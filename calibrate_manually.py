#!/usr/bin/env python3
# calibrate_thermistors.py
import sys
import time
import glob
import math
import csv
from datetime import datetime

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QMessageBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView
)

import pyqtgraph as pg
from gpiozero.pins.rpigpio import RPiGPIOFactory
from gpiozero import PWMOutputDevice, OutputDevice
import serial


# ============================================================
# CONFIG (EDIT THESE ONLY)
# ============================================================

# ---- Read/update frequency ----
READ_LOOP_HZ = 5.0               # how fast we read Pico + refresh plot

# PWM frequency for heater output (SSR recommended; do NOT use with a mechanical relay)
HEATER_PWM_HZ = 30.0

# ---- GPIO pins (BCM) ----
HEATER_GPIO_BCM = 18             # heater SSR/PWM pin
FANS_GPIO_BCM = 21               # fan relay pin (ON/OFF)

# Fan relay polarity
FANS_RELAY_ACTIVE_HIGH = True    # set False if relay turns ON when GPIO is LOW

# ---- Temperature safety limits (°C) ----
TEMP_MIN_C = -20.0
TEMP_MAX_C = 120.0

# ---- Pico serial ----
PICO_BAUD = 115200
PICO_READ_TIMEOUT_S = 0.2        # serial timeout

# ---- ADC conversion ----
VREF = 3.3
ADC_MAX = 65535

# ---- Voltage divider / thermistor model ----
R_FIXED_OHM = 10_000.0           # fixed resistor in the divider (ohms)
NTC_R25_OHM = 10_000.0           # thermistor resistance at 25C (ohms)
NTC_BETA_K = 3950.0              # Beta (K)
NTC_T0_K = 25.0 + 273.15         # 25C in Kelvin

# Wiring assumption:
# True  => 3.3V -> R_FIXED -> ADC node -> NTC -> GND
# False => 3.3V -> NTC -> ADC node -> R_FIXED -> GND
DIVIDER_PULLUP = True

# Which signals to log when you click "Add calibration point"
# Average this many most recent *filtered* samples per sensor
POINT_AVG_N = 5

# ---- Low-pass filter for thermistor temperatures ----
# Reasonable default: 5 s time constant.
# At READ_LOOP_HZ=5, dt≈0.2s -> alpha≈0.2/(5.0+0.2)=0.038
TEMP_LP_TAU_S = 5.0              # seconds (set 2..10 typically)

# Default CSV output name
DEFAULT_CSV_PREFIX = "thermistor_calibration"


# ============================================================
# Pico serial helpers
# ============================================================
def find_pico_port() -> str:
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    raise FileNotFoundError("No /dev/ttyACM* or /dev/ttyUSB* device found. Is the Pico plugged in?")


def parse_pico_line(line: str):
    parts = line.strip().split(",")
    if len(parts) != 4:
        raise ValueError(f"Bad column count: {parts}")
    t_ms = int(parts[0])
    r0 = int(parts[1]); r1 = int(parts[2]); r2 = int(parts[3])
    return t_ms, r0, r1, r2


def adc_to_volts(r: int) -> float:
    r = max(0, min(int(r), ADC_MAX))
    return (r / ADC_MAX) * VREF


def volts_to_r_therm(v: float) -> float:
    v = float(v)
    v = max(1e-6, min(VREF - 1e-6, v))

    if DIVIDER_PULLUP:
        # Vout = Vref * (R_th / (R_fixed + R_th))  =>  R_th = R_fixed * Vout / (Vref - Vout)
        return R_FIXED_OHM * v / (VREF - v)
    else:
        # Vout = Vref * (R_fixed / (R_fixed + R_th))  =>  R_th = R_fixed * (Vref - Vout) / Vout
        return R_FIXED_OHM * (VREF - v) / v


def r_therm_to_celsius(r_th: float) -> float:
    r_th = max(1e-3, float(r_th))
    inv_T = (1.0 / NTC_T0_K) + (1.0 / NTC_BETA_K) * math.log(r_th / NTC_R25_OHM)
    T = 1.0 / inv_T
    return T - 273.15


def adc_to_temp_c(adc: int) -> float:
    v = adc_to_volts(adc)
    r_th = volts_to_r_therm(v)
    return r_therm_to_celsius(r_th)


# ============================================================
# Low-pass filter helper
# ============================================================
def lp_alpha(dt: float, tau: float) -> float:
    dt = max(1e-6, float(dt))
    tau = max(1e-6, float(tau))
    return dt / (tau + dt)


# ============================================================
# Worker thread: reads Pico + applies manual PWM duty
# ============================================================
class CalibThread(QThread):
    update = Signal(dict)
    fault = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._force_off = False
        self._duty = 0.0  # 0..1

        self.heater = PWMOutputDevice(
            HEATER_GPIO_BCM,
            active_high=True,
            initial_value=0.0,
            frequency=HEATER_PWM_HZ,
            pin_factory=RPiGPIOFactory()
        )

        self._ser = None
        self._last_raw = [float("nan"), float("nan"), float("nan")]

        # Filter state (temps in °C)
        self._lp = [float("nan"), float("nan"), float("nan")]
        self._last_update_t = None  # wall time (s)

    def stop(self):
        self._running = False

    def force_heater_off(self):
        self._force_off = True
        try:
            self.heater.value = 0.0
            self.heater.off()
        except Exception:
            pass

    def clear_force_off(self):
        self._force_off = False

    def set_duty_percent(self, duty_percent: float):
        d = float(duty_percent) / 100.0
        self._duty = max(0.0, min(1.0, d))

    def _open_pico(self):
        port = find_pico_port()
        self._ser = serial.Serial(port, baudrate=PICO_BAUD, timeout=PICO_READ_TIMEOUT_S)
        self._ser.reset_input_buffer()

    def _read_latest_raw_temps(self):
        if self._ser is None:
            return self._last_raw

        # Drain a few lines so we use the most recent sample
        for _ in range(30):
            raw = self._ser.readline()
            if not raw:
                break

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("Pico ADC started"):
                continue

            try:
                _, r0, r1, r2 = parse_pico_line(line)
                t0 = adc_to_temp_c(r0)
                t1 = adc_to_temp_c(r1)
                t2 = adc_to_temp_c(r2)
                self._last_raw = [t0, t1, t2]
            except Exception:
                continue

        return self._last_raw

    def _update_lowpass(self, x, now_t: float):
        """
        First-order IIR LPF applied per channel in °C.
        Handles NaNs gracefully: if raw is NaN -> keep previous;
        if previous is NaN and raw is valid -> initialize to raw.
        """
        if self._last_update_t is None:
            self._last_update_t = now_t

        dt = max(1e-6, now_t - self._last_update_t)
        a = lp_alpha(dt, TEMP_LP_TAU_S)

        y = self._lp[:]
        for i in range(3):
            xi = float(x[i])
            yi = float(y[i])

            if math.isnan(xi):
                # no new info -> keep old filtered value
                continue

            if math.isnan(yi):
                # initialize filter when first valid reading arrives
                y[i] = xi
            else:
                y[i] = yi + a * (xi - yi)

        self._lp = y
        self._last_update_t = now_t
        return self._lp

    def run(self):
        wall_t0 = time.time()
        try:
            self._open_pico()

            while self._running:
                loop_start = time.time()
                t_s = loop_start - wall_t0

                raw_temps = self._read_latest_raw_temps()
                filt_temps = self._update_lowpass(raw_temps, loop_start)

                # Safety check on filtered temps (more robust than raw spikes)
                for ti in filt_temps:
                    if not math.isnan(ti) and (ti > TEMP_MAX_C or ti < TEMP_MIN_C):
                        self.force_heater_off()
                        self.fault.emit(f"Temperature out of bounds: {ti:.2f} °C (heater forced OFF)")
                        return

                if self._force_off:
                    self.heater.value = 0.0
                    applied = 0.0
                else:
                    applied = float(max(0.0, min(1.0, self._duty)))
                    self.heater.value = applied

                self.update.emit({
                    "t_s": t_s,
                    "temps_c": list(filt_temps),     # <-- filtered temps are what GUI/logging uses
                    "temps_raw_c": list(raw_temps),  # optional, if you ever want to display it
                    "duty": applied,
                    "forced_off": self._force_off,
                    "lp_tau_s": TEMP_LP_TAU_S,
                })

                period = 1.0 / max(0.1, READ_LOOP_HZ)
                time.sleep(max(0.0, period - (time.time() - loop_start)))

        except Exception as e:
            self.force_heater_off()
            self.fault.emit(str(e))
        finally:
            try:
                self.force_heater_off()
            except Exception:
                pass
            try:
                if self._ser is not None:
                    self._ser.close()
            except Exception:
                pass


# ============================================================
# GUI: manual PWM + add calibration points to CSV table
# ============================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thermistor Calibration (Manual PWM + Reference Temp Logging)")

        # Fan relay device (optional)
        self.fans = OutputDevice(FANS_GPIO_BCM, active_high=FANS_RELAY_ACTIVE_HIGH, initial_value=False)
        self.fans_on = False

        # CSV output
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"{DEFAULT_CSV_PREFIX}_{ts}.csv"
        self._csv_file = None
        self._csv_writer = None

        # Latest temps buffer for averaging (FILTERED temps)
        self.latest_samples = []  # list of [t0,t1,t2]

        # Worker thread
        self.worker = None

        # ---------------- UI controls ----------------
        self.btn_start = QPushButton("Start Calibration")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_emergency_off = QPushButton("⛔ EMERGENCY HEATER OFF")
        self.btn_emergency_off.setMinimumHeight(60)
        self.btn_emergency_off.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: 800;
                background-color: #b00020;
                color: white;
                border-radius: 12px;
                padding: 10px;
            }
            QPushButton:pressed { background-color: #7a0016; }
        """)

        self.btn_fans = QPushButton("Fans: OFF")
        self.btn_fans.setMinimumHeight(40)

        self.duty = QDoubleSpinBox()
        self.duty.setRange(0.0, 100.0)
        self.duty.setDecimals(1)
        self.duty.setSingleStep(1.0)
        self.duty.setValue(0.0)
        self.duty.setSuffix(" %")

        self.btn_apply_duty = QPushButton("Apply Duty")

        self.real_temp = QDoubleSpinBox()
        self.real_temp.setRange(-50.0, 300.0)
        self.real_temp.setDecimals(2)
        self.real_temp.setSingleStep(0.5)
        self.real_temp.setValue(25.0)
        self.real_temp.setSuffix(" °C")

        self.btn_add_point = QPushButton("Add calibration point")
        self.btn_add_point.setEnabled(False)

        self.lbl_status = QLabel(
            f"Status: idle | heaterGPIO={HEATER_GPIO_BCM} | fansGPIO={FANS_GPIO_BCM} | "
            f"PWM={HEATER_PWM_HZ}Hz | LPF tau={TEMP_LP_TAU_S}s | log={self.csv_path}"
        )
        self.lbl_now = QLabel("t=-- s | duty=-- % | T0/T1/T2=--/--/-- °C")

        # ---------------- Table ----------------
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Real temp (°C)", "Thermometer 0 (°C)", "Thermometer 1 (°C)", "Thermometer 2 (°C)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # ---------------- Plot ----------------
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setLabel("left", "Temperature (°C)")
        self.plot.addLegend()

        self.curve_t0 = self.plot.plot([], [], name="Temp sensor0 (LPF)", pen=pg.mkPen(width=1))
        self.curve_t1 = self.plot.plot([], [], name="Temp sensor1 (LPF)", pen=pg.mkPen(width=1))
        self.curve_t2 = self.plot.plot([], [], name="Temp sensor2 (LPF)", pen=pg.mkPen(width=1))

        self.hist_t = []
        self.hist_t0 = []
        self.hist_t1 = []
        self.hist_t2 = []

        # ---------------- Layout ----------------
        top = QVBoxLayout()
        top.addWidget(self.btn_emergency_off)
        top.addWidget(self.btn_fans)

        g = QGroupBox("Manual Heater PWM")
        gl = QVBoxLayout()
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Duty:"))
        r1.addWidget(self.duty)
        r1.addWidget(self.btn_apply_duty)
        gl.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Reference (real) temp:"))
        r2.addWidget(self.real_temp)
        r2.addWidget(self.btn_add_point)
        gl.addLayout(r2)
        g.setLayout(gl)
        top.addWidget(g)

        runrow = QHBoxLayout()
        runrow.addWidget(self.btn_start)
        runrow.addWidget(self.btn_stop)
        top.addLayout(runrow)

        top.addWidget(self.lbl_status)
        top.addWidget(self.lbl_now)
        top.addWidget(self.plot)
        top.addWidget(QLabel("Calibration look-up table (saved to CSV as you add points):"))
        top.addWidget(self.table)
        self.setLayout(top)

        # Wiring
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_apply_duty.clicked.connect(self.on_apply_duty)
        self.btn_add_point.clicked.connect(self.on_add_point)
        self.btn_emergency_off.clicked.connect(self.on_emergency_off)
        self.btn_fans.clicked.connect(self.on_toggle_fans)

    # ---------- helpers ----------
    def _set_fans_ui(self, on: bool):
        self.fans_on = bool(on)
        if self.fans_on:
            self.btn_fans.setText("Fans: ON")
            self.btn_fans.setStyleSheet("font-weight: 700;")
        else:
            self.btn_fans.setText("Fans: OFF")
            self.btn_fans.setStyleSheet("")

    def _open_csv(self):
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["real_temp_c", "thermometer0_c", "thermometer1_c", "thermometer2_c"])

    def _close_csv(self):
        try:
            if self._csv_file is not None:
                self._csv_file.close()
        except Exception:
            pass
        self._csv_file = None
        self._csv_writer = None

    def _avg_latest(self):
        if not self.latest_samples:
            return [float("nan"), float("nan"), float("nan")]
        n = min(POINT_AVG_N, len(self.latest_samples))
        chunk = self.latest_samples[-n:]
        out = []
        for j in range(3):
            vals = [row[j] for row in chunk if not math.isnan(row[j])]
            out.append(sum(vals) / len(vals) if vals else float("nan"))
        return out

    # ---------- slots ----------
    def on_toggle_fans(self):
        try:
            if self.fans_on:
                self.fans.off()
                self._set_fans_ui(False)
                self.lbl_status.setText("Status: fans OFF")
            else:
                self.fans.on()
                self._set_fans_ui(True)
                self.lbl_status.setText("Status: fans ON")
        except Exception as e:
            QMessageBox.critical(self, "Fan toggle failed", str(e))

    def on_start(self):
        if self.worker is not None:
            return

        try:
            self._open_csv()
        except Exception as e:
            QMessageBox.critical(self, "CSV open failed", str(e))
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_add_point.setEnabled(True)

        self.latest_samples.clear()
        self.hist_t.clear(); self.hist_t0.clear(); self.hist_t1.clear(); self.hist_t2.clear()
        self.curve_t0.setData([], []); self.curve_t1.setData([], []); self.curve_t2.setData([], [])

        self.worker = CalibThread()
        self.worker.update.connect(self.on_update)
        self.worker.fault.connect(self.on_fault)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

        # Apply initial duty immediately
        self.on_apply_duty()

        self.lbl_status.setText(
            f"Status: running | logging to {self.csv_path} | LPF tau={TEMP_LP_TAU_S}s"
        )

    def on_stop(self):
        if self.worker is not None:
            self.worker.force_heater_off()
            self.worker.stop()
        self.lbl_status.setText("Status: stopping...")

    def on_emergency_off(self):
        if self.worker is not None:
            self.worker.force_heater_off()
        self.lbl_status.setText("Status: EMERGENCY OFF pressed — heater forced OFF")

    def on_apply_duty(self):
        if self.worker is None:
            return
        try:
            self.worker.clear_force_off()
            self.worker.set_duty_percent(float(self.duty.value()))
            self.lbl_status.setText(
                f"Status: duty applied = {self.duty.value():.1f}% | LPF tau={TEMP_LP_TAU_S}s | logging to {self.csv_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Apply duty failed", str(e))

    def on_add_point(self):
        if self.worker is None:
            QMessageBox.warning(self, "Not running", "Start calibration first.")
            return

        temps = self._avg_latest()
        if any(math.isnan(x) for x in temps):
            QMessageBox.warning(self, "No sensor data", "Waiting for valid sensor readings from the Pico.")
            return

        real_c = float(self.real_temp.value())
        t0, t1, t2 = temps

        # Append to table UI
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate([real_c, t0, t1, t2]):
            it = QTableWidgetItem(f"{val:.3f}")
            self.table.setItem(row, col, it)

        # Write to CSV
        try:
            if self._csv_writer is not None:
                self._csv_writer.writerow([f"{real_c:.6f}", f"{t0:.6f}", f"{t1:.6f}", f"{t2:.6f}"])
                self._csv_file.flush()
        except Exception as e:
            QMessageBox.critical(self, "CSV write failed", str(e))

    def on_update(self, d: dict):
        t_s = float(d.get("t_s", 0.0))
        temps = d.get("temps_c", [float("nan")]*3)  # filtered temps
        t0, t1, t2 = (float(temps[0]), float(temps[1]), float(temps[2]))
        duty = float(d.get("duty", 0.0)) * 100.0
        forced = bool(d.get("forced_off", False))

        # keep recent filtered samples for averaging a calibration point
        self.latest_samples.append([t0, t1, t2])
        if len(self.latest_samples) > 300:
            self.latest_samples = self.latest_samples[-300:]

        # plot history
        self.hist_t.append(t_s)
        self.hist_t0.append(t0)
        self.hist_t1.append(t1)
        self.hist_t2.append(t2)

        self.curve_t0.setData(self.hist_t, self.hist_t0)
        self.curve_t1.setData(self.hist_t, self.hist_t1)
        self.curve_t2.setData(self.hist_t, self.hist_t2)

        heater_txt = "FORCED OFF" if forced else "PWM"
        self.lbl_now.setText(
            f"t={t_s:7.1f} s | duty={duty:5.1f}% | heater={heater_txt} | "
            f"T0/T1/T2={t0:6.2f}/{t1:6.2f}/{t2:6.2f} °C"
        )

    def on_fault(self, msg: str):
        QMessageBox.critical(self, "Fault", msg)
        self.lbl_status.setText(f"Status: fault — {msg}")
        self.on_stop()

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_add_point.setEnabled(False)
        self.worker = None
        self._close_csv()
        self.lbl_status.setText(f"Status: stopped | saved {self.csv_path}")

    def closeEvent(self, event):
        # Ensure fans OFF on exit
        try:
            self.fans.off()
        except Exception:
            pass

        if self.worker is not None:
            try:
                self.worker.force_heater_off()
            except Exception:
                pass
            self.worker.stop()
            self.worker.wait(2000)

        self._close_csv()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1000, 750)
    w.show()
    sys.exit(app.exec())
