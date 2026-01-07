# run_oven_pico.py
import sys
import time
import glob
import math
import csv
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QMessageBox, QGroupBox
)

import pyqtgraph as pg
from gpiozero.pins.rpigpio import RPiGPIOFactory
from gpiozero import PWMOutputDevice, OutputDevice
import serial


# ============================================================
# CONFIG (EDIT THESE ONLY)
# ============================================================

# ---- Frequencies / timing ----
CONTROL_LOOP_HZ = 30.0            # sensor read + PID compute frequency (Hz)
PROFILE_MAP_HZ = 1.0             # setpoint map resolution (Hz)

# PWM frequency for heater output (SSR recommended; do NOT use with a mechanical relay)
HEATER_PWM_HZ = 30.0

# ---- GPIO pins (BCM) ----
HEATER_GPIO_BCM = 18             # heater SSR/PWM pin
FANS_GPIO_BCM = 21               # fan relay pin (ON/OFF)

# Fan relay polarity
FANS_RELAY_ACTIVE_HIGH = True    # set False if relay turns ON when GPIO is LOW

# ---- Temperature safety limits (°C) ----
TEMP_MIN_C = 0.0
TEMP_MAX_C = 140.0

# ---- PID control parameters ----
PID_KP = 0.22                     # proportional gain
PID_KI = 0.03                    # integral gain (per second)
PID_KD = 0.0                     # derivative gain (per second)

PID_OUTPUT_MIN = 0.0             # interpreted as duty request [0..1]
PID_OUTPUT_MAX = 1.0

INTEGRAL_MIN = -2.0              # anti-windup clamp (in "output-equivalent" units)
INTEGRAL_MAX = 2.0

# Profile quantization (setpoint accuracy)
SETPOINT_RES_C = 0.1

# ---- Initial heat-up stage ----
INITIAL_BAND_C = 0.2             # °C tolerance for "initial temp reached"

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

# Which sensor drives the controller PV (0,1,2)
CONTROL_SENSOR_INDEX = 0

# ---- Calibration LUT CSV ----
# CSV columns expected: real_temp_c, thermometer0_c, thermometer1_c, thermometer2_c
# Put your calibration file path here (or keep as relative if you run from same folder).
CALIBRATION_CSV_PATH = "thermistor_calibration_20251230_184921.csv"

# If True: if LUT missing/unreadable, fall back to uncalibrated temps (still low-pass filtered).
ALLOW_UNCALIBRATED_FALLBACK = True

# ---- Low-pass filter for temperature readings (same as calibration tool) ----
TEMP_LP_TAU_S = 5.0              # seconds (2..10 typical)


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
# Calibration LUT + interpolation
# ============================================================
def _interp_extrap(x: float, xp: list[float], yp: list[float]) -> float:
    """
    Linear interpolation y(x) with linear extrapolation outside endpoints.
    xp must be sorted ascending and have at least 2 points.
    """
    if not xp or not yp or len(xp) != len(yp) or len(xp) < 2:
        return float("nan")

    # Extrapolate below range using first segment
    if x <= xp[0]:
        x0, x1 = xp[0], xp[1]
        y0, y1 = yp[0], yp[1]
        dx = x1 - x0
        if abs(dx) < 1e-12:
            return y0
        return y0 + (x - x0) * (y1 - y0) / dx

    # Extrapolate above range using last segment
    if x >= xp[-1]:
        x0, x1 = xp[-2], xp[-1]
        y0, y1 = yp[-2], yp[-1]
        dx = x1 - x0
        if abs(dx) < 1e-12:
            return y1
        return y1 + (x - x1) * (y1 - y0) / dx

    # Interpolate inside range
    lo = 0
    hi = len(xp) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x < xp[mid]:
            hi = mid
        else:
            lo = mid

    x0, x1 = xp[lo], xp[hi]
    y0, y1 = yp[lo], yp[hi]
    dx = x1 - x0
    if abs(dx) < 1e-12:
        return y0
    t = (x - x0) / dx
    return y0 + t * (y1 - y0)



class ThermistorCalibrator:
    """
    Builds per-sensor mapping: measured thermistor temp -> real temp.
    For each sensor i: xp_i = sorted(measured temps), yp_i = real temps.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.ok = False
        self.note = ""
        self.xp = [[], [], []]  # measured temps
        self.yp = [[], [], []]  # real temps

        self._load()

    def _load(self):
        try:
            rows = []
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                # Expect these headers from your calibration script:
                # real_temp_c, thermometer0_c, thermometer1_c, thermometer2_c
                required = {"real_temp_c", "thermometer0_c", "thermometer1_c", "thermometer2_c"}
                if not reader.fieldnames:
                    raise ValueError("CSV has no header row.")
                missing = required - set(reader.fieldnames)
                if missing:
                    raise ValueError(f"CSV missing columns: {sorted(missing)}")

                for r in reader:
                    try:
                        real = float(r["real_temp_c"])
                        t0 = float(r["thermometer0_c"])
                        t1 = float(r["thermometer1_c"])
                        t2 = float(r["thermometer2_c"])
                        if any(math.isnan(v) for v in (real, t0, t1, t2)):
                            continue
                        rows.append((real, t0, t1, t2))
                    except Exception:
                        continue

            if len(rows) < 2:
                raise ValueError("Need at least 2 calibration rows to interpolate.")

            # Build mapping per sensor: sort by measured temperature
            for i in range(3):
                pairs = []
                for (real, t0, t1, t2) in rows:
                    meas = [t0, t1, t2][i]
                    pairs.append((meas, real))

                # sort by meas
                pairs.sort(key=lambda p: p[0])

                # collapse duplicate meas by averaging real
                collapsed_x = []
                collapsed_y = []
                k = 0
                while k < len(pairs):
                    xk = pairs[k][0]
                    ys = [pairs[k][1]]
                    k += 1
                    while k < len(pairs) and abs(pairs[k][0] - xk) < 1e-9:
                        ys.append(pairs[k][1])
                        k += 1
                    collapsed_x.append(xk)
                    collapsed_y.append(sum(ys) / len(ys))

                self.xp[i] = collapsed_x
                self.yp[i] = collapsed_y

            self.ok = True
            self.note = f"Calibration loaded from {self.csv_path}"
        except Exception as e:
            self.ok = False
            self.note = f"Calibration load failed: {e}"

    def calibrate(self, meas_temp_c: float, sensor_index: int) -> float:
        if math.isnan(meas_temp_c):
            return float("nan")
        if not self.ok:
            return meas_temp_c
        i = int(sensor_index)
        return _interp_extrap(meas_temp_c, self.xp[i], self.yp[i])



# ============================================================
# Low-pass filter helper
# ============================================================
def lp_alpha(dt: float, tau: float) -> float:
    dt = max(1e-6, float(dt))
    tau = max(1e-6, float(tau))
    return dt / (tau + dt)


# ============================================================
# Profile generation
# ============================================================
@dataclass
class Profile:
    t_s: list[int]
    sp_c: list[float]
    total_s: int


def quantize_temp(x: float) -> float:
    return round(x / SETPOINT_RES_C) * SETPOINT_RES_C


def build_profile(
    start_temp_c: float,
    ramp_c_per_min: float,
    hold_temp_c: float,
    hold_time_min: float,
    soak_c_per_min: float,
    end_temp_c: float,
) -> Profile:
    if ramp_c_per_min <= 0:
        raise ValueError("Ramp rate must be > 0")
    if soak_c_per_min <= 0:
        raise ValueError("Soak rate must be > 0")
    if hold_time_min < 0:
        raise ValueError("Hold time must be >= 0")
    if PROFILE_MAP_HZ <= 0:
        raise ValueError("PROFILE_MAP_HZ must be > 0")

    dt = 1.0 / PROFILE_MAP_HZ
    ramp_c_per_s = ramp_c_per_min / 60.0
    soak_c_per_s = soak_c_per_min / 60.0

    t_s: list[int] = []
    sp: list[float] = []

    # Phase 1: ramp
    temp = start_temp_c
    direction = 1.0 if hold_temp_c >= start_temp_c else -1.0
    rate = ramp_c_per_s * direction

    t = 0.0
    while True:
        t_s.append(int(round(t)))
        sp.append(quantize_temp(temp))

        if (direction > 0 and temp >= hold_temp_c) or (direction < 0 and temp <= hold_temp_c):
            break

        temp += rate * dt
        t += dt

        if t > 24 * 3600:
            raise ValueError("Ramp phase exceeded 24 hours — check rates/targets")

    sp[-1] = quantize_temp(hold_temp_c)
    temp = hold_temp_c

    # Phase 2: hold
    hold_s = int(round(hold_time_min * 60))
    for _ in range(hold_s):
        t += 1.0
        t_s.append(int(round(t)))
        sp.append(quantize_temp(temp))

    # Phase 3: soak
    direction2 = 1.0 if end_temp_c >= temp else -1.0
    rate2 = soak_c_per_s * direction2

    while True:
        if (direction2 > 0 and temp >= end_temp_c) or (direction2 < 0 and temp <= end_temp_c):
            break
        temp += rate2 * dt
        t += dt
        t_s.append(int(round(t)))
        sp.append(quantize_temp(temp))

        if t > 24 * 3600:
            raise ValueError("Soak phase exceeded 24 hours — check rates/targets")

    if sp:
        sp[-1] = quantize_temp(end_temp_c)

    total_s = t_s[-1] if t_s else 0
    return Profile(t_s=t_s, sp_c=sp, total_s=total_s)


# ============================================================
# PID controller
# ============================================================
class PID:
    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float, out_max: float,
                 i_min: float, i_max: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.i_min = i_min
        self.i_max = i_max

        self.integral = 0.0
        self.prev_err = 0.0
        self.prev_t = None

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
        self.prev_t = None

    def update(self, setpoint: float, measurement: float, t_now: float) -> float:
        if self.prev_t is None:
            self.prev_t = t_now
            self.prev_err = setpoint - measurement
            return 0.0

        dt = max(1e-6, t_now - self.prev_t)
        err = setpoint - measurement

        self.integral += err * dt
        self.integral = max(self.i_min, min(self.integral, self.i_max))

        derr = (err - self.prev_err) / dt

        u = (self.kp * err) + (self.ki * self.integral) + (self.kd * derr)
        u = max(self.out_min, min(u, self.out_max))

        self.prev_err = err
        self.prev_t = t_now
        return u


# ============================================================
# Control thread (reads Pico + drives heater via PWM)
# ============================================================
class ControlThread(QThread):
    update = Signal(dict)
    fault = Signal(str)

    def __init__(self, profile: Profile, initial_target_c: float, parent=None):
        super().__init__(parent)
        self.profile = profile
        self.initial_target_c = float(initial_target_c)
        self._running = True
        self._force_off = False

        # PWM output: value in [0..1]
        self.heater = PWMOutputDevice(
            HEATER_GPIO_BCM,
            active_high=True,
            initial_value=0.0,
            frequency=HEATER_PWM_HZ,
            pin_factory=RPiGPIOFactory()
        )

        self.pid = PID(
            kp=PID_KP, ki=PID_KI, kd=PID_KD,
            out_min=PID_OUTPUT_MIN, out_max=PID_OUTPUT_MAX,
            i_min=INTEGRAL_MIN, i_max=INTEGRAL_MAX
        )

        self._ser = None

        # Calibration
        self.cal = ThermistorCalibrator(CALIBRATION_CSV_PATH)
        if (not self.cal.ok) and (not ALLOW_UNCALIBRATED_FALLBACK):
            raise RuntimeError(self.cal.note)

        # Filter state (on calibrated temps)
        self._lp = [float("nan"), float("nan"), float("nan")]
        self._lp_last_t = None  # wall time

        # last output temps to GUI (filtered, calibrated)
        self._last_temps = [float("nan"), float("nan"), float("nan")]

    def stop(self):
        self._running = False

    def force_heater_off(self):
        self._force_off = True
        try:
            self.heater.value = 0.0
            self.heater.off()
        except Exception:
            pass

    def _open_pico(self):
        port = find_pico_port()
        self._ser = serial.Serial(port, baudrate=PICO_BAUD, timeout=PICO_READ_TIMEOUT_S)
        self._ser.reset_input_buffer()

    def _apply_lowpass(self, x_cal, now_t: float):
        if self._lp_last_t is None:
            self._lp_last_t = now_t
        dt = max(1e-6, now_t - self._lp_last_t)
        a = lp_alpha(dt, TEMP_LP_TAU_S)

        y = self._lp[:]
        for i in range(3):
            xi = float(x_cal[i])
            yi = float(y[i])

            if math.isnan(xi):
                continue
            if math.isnan(yi):
                y[i] = xi
            else:
                y[i] = yi + a * (xi - yi)

        self._lp = y
        self._lp_last_t = now_t
        return self._lp

    def _read_latest_temps(self):
        """
        Returns temps after:
          ADC->temp (model) -> calibration LUT (interp) -> low-pass filter
        """
        if self._ser is None:
            return self._last_temps

        got_any = False
        last_raw_model = None

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
                t0m = adc_to_temp_c(r0)
                t1m = adc_to_temp_c(r1)
                t2m = adc_to_temp_c(r2)
                last_raw_model = [t0m, t1m, t2m]
                got_any = True
            except Exception:
                continue

        if not got_any or last_raw_model is None:
            return self._last_temps

        # calibration (measured->real)
        t_cal = [
            self.cal.calibrate(last_raw_model[0], 0),
            self.cal.calibrate(last_raw_model[1], 1),
            self.cal.calibrate(last_raw_model[2], 2),
        ]

        # low-pass filter on calibrated temps
        now_t = time.time()
        t_filt = self._apply_lowpass(t_cal, now_t)

        self._last_temps = list(t_filt)
        return self._last_temps

    def _get_average_pv(self, temps):
        """Helper to calculate average of available sensors for PID control"""
        valid_temps = [t for t in temps if not math.isnan(t)]
        if not valid_temps:
            return float("nan")
        return sum(valid_temps) / len(valid_temps)

    def run(self):
        wall_t0 = time.time()
        self.pid.reset()

        try:
            self._open_pico()

            # -------------------------
            # Phase A: initial heat-up
            # -------------------------
            reached_initial = False
            while self._running and not reached_initial:
                loop_start = time.time()
                total_elapsed_s = int(loop_start - wall_t0)

                temps = self._read_latest_temps()
                # CHANGE: Use average instead of CONTROL_SENSOR_INDEX
                pv = self._get_average_pv(temps)

                if math.isnan(pv):
                    self.update.emit({
                        "stage": "initial",
                        "t_total_s": total_elapsed_s,
                        "t_stage_s": total_elapsed_s,
                        "setpoint_c": self.initial_target_c,
                        "actual_c": float("nan"),
                        "temps_c": temps,
                        "forced_off": self._force_off,
                        "duty": 0.0,
                        "done": False,
                        "note": f"Waiting for Pico data... | {self.cal.note}"
                    })
                    time.sleep(0.1)
                    continue

                if pv > TEMP_MAX_C or pv < TEMP_MIN_C:
                    self.force_heater_off()
                    self.fault.emit(f"Average temp out of bounds: {pv:.2f} °C")
                    return

                if self._force_off:
                    duty = 0.0
                else:
                    duty = self.pid.update(setpoint=self.initial_target_c, measurement=pv, t_now=loop_start)

                if self._force_off:
                    self.heater.value = 0.0
                else:
                    self.heater.value = float(max(0.0, min(1.0, duty)))

                reached_initial = (pv >= (self.initial_target_c - INITIAL_BAND_C))

                self.update.emit({
                    "stage": "initial",
                    "t_total_s": total_elapsed_s,
                    "t_stage_s": total_elapsed_s,
                    "setpoint_c": self.initial_target_c,
                    "actual_c": pv,
                    "temps_c": temps,
                    "forced_off": self._force_off,
                    "duty": duty,
                    "done": False,
                    "note": f"{self.cal.note} (using AVG of sensors)"
                })

                period = 1.0 / CONTROL_LOOP_HZ
                time.sleep(max(0.0, period - (time.time() - loop_start)))

            if not self._running:
                self.force_heater_off()
                return

            self.pid.reset()

            # -------------------------
            # Phase B: main profile
            # -------------------------
            profile_start = time.time()
            idx = 0

            while self._running:
                loop_start = time.time()
                total_elapsed_s = int(loop_start - wall_t0)
                elapsed_profile_s = int(loop_start - profile_start)

                while (idx + 1 < len(self.profile.t_s)) and (elapsed_profile_s >= self.profile.t_s[idx + 1]):
                    idx += 1

                sp = self.profile.sp_c[idx] if self.profile.sp_c else 0.0

                temps = self._read_latest_temps()
                # CHANGE: Use average instead of CONTROL_SENSOR_INDEX
                pv = self._get_average_pv(temps)

                if math.isnan(pv):
                    self.force_heater_off()
                    self.fault.emit("Lost temperature data (Average is NaN)")
                    break

                if pv > TEMP_MAX_C or pv < TEMP_MIN_C:
                    self.force_heater_off()
                    self.fault.emit(f"Average temp out of bounds: {pv:.2f} °C")
                    break

                if self._force_off:
                    duty = 0.0
                else:
                    duty = self.pid.update(setpoint=sp, measurement=pv, t_now=loop_start)

                if self._force_off:
                    self.heater.value = 0.0
                else:
                    self.heater.value = float(max(0.0, min(1.0, duty)))

                done = (elapsed_profile_s >= self.profile.total_s)

                self.update.emit({
                    "stage": "profile",
                    "t_total_s": total_elapsed_s,
                    "t_stage_s": elapsed_profile_s,
                    "setpoint_c": sp,
                    "actual_c": pv,
                    "temps_c": temps,
                    "forced_off": self._force_off,
                    "duty": duty,
                    "done": done,
                    "note": f"{self.cal.note} (using AVG of sensors)"
                })

                if done:
                    self.force_heater_off()
                    break

                period = 1.0 / CONTROL_LOOP_HZ
                time.sleep(max(0.0, period - (time.time() - loop_start)))

            if not self._running:
                self.force_heater_off()
                return

            # Reset PID before main cycle
            self.pid.reset()

            # -------------------------
            # Phase B: main profile
            # -------------------------
            profile_start = time.time()
            idx = 0

            while self._running:
                loop_start = time.time()
                total_elapsed_s = int(loop_start - wall_t0)
                elapsed_profile_s = int(loop_start - profile_start)

                while (idx + 1 < len(self.profile.t_s)) and (elapsed_profile_s >= self.profile.t_s[idx + 1]):
                    idx += 1

                sp = self.profile.sp_c[idx] if self.profile.sp_c else 0.0

                temps = self._read_latest_temps()
                pv = float(temps[CONTROL_SENSOR_INDEX])

                if math.isnan(pv):
                    self.force_heater_off()
                    self.fault.emit("Lost Pico temperature data (PV is NaN) — heater forced OFF")
                    break

                if pv > TEMP_MAX_C or pv < TEMP_MIN_C:
                    self.force_heater_off()
                    self.fault.emit(f"Temperature out of bounds: {pv:.2f} °C (heater forced OFF)")
                    break

                if self._force_off:
                    duty = 0.0
                else:
                    duty = self.pid.update(setpoint=sp, measurement=pv, t_now=loop_start)

                if self._force_off:
                    self.heater.value = 0.0
                else:
                    self.heater.value = float(max(0.0, min(1.0, duty)))

                done = (elapsed_profile_s >= self.profile.total_s)

                self.update.emit({
                    "stage": "profile",
                    "t_total_s": total_elapsed_s,
                    "t_stage_s": elapsed_profile_s,
                    "setpoint_c": sp,
                    "actual_c": pv,
                    "temps_c": temps,
                    "forced_off": self._force_off,
                    "duty": duty,
                    "done": done,
                    "note": f"{self.cal.note} | LPF tau={TEMP_LP_TAU_S}s"
                })

                if done:
                    self.force_heater_off()
                    break

                period = 1.0 / CONTROL_LOOP_HZ
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
# GUI
# ============================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pi Temperature Profile Controller (Pico ADC NTC + PID + PWM)")

        # Fan relay device (ON/OFF)
        self.fans = OutputDevice(FANS_GPIO_BCM, active_high=FANS_RELAY_ACTIVE_HIGH, initial_value=False)
        self.fans_on = False

        # Inputs
        self.initial_temp = QDoubleSpinBox()
        self.initial_temp.setRange(-50.0, 200.0)
        self.initial_temp.setValue(25.0)
        self.initial_temp.setSuffix(" °C")

        self.ramp = QDoubleSpinBox(); self.ramp.setRange(0.1, 200.0); self.ramp.setValue(10.0); self.ramp.setSuffix(" °C/min")
        self.hold_temp = QDoubleSpinBox(); self.hold_temp.setRange(-50.0, 200.0); self.hold_temp.setValue(60.0); self.hold_temp.setSuffix(" °C")
        self.hold_time = QDoubleSpinBox(); self.hold_time.setRange(0.0, 600.0); self.hold_time.setValue(10.0); self.hold_time.setSuffix(" min")
        self.soak = QDoubleSpinBox(); self.soak.setRange(0.1, 200.0); self.soak.setValue(10.0); self.soak.setSuffix(" °C/min")
        self.end_temp = QDoubleSpinBox(); self.end_temp.setRange(-50.0, 200.0); self.end_temp.setValue(25.0); self.end_temp.setSuffix(" °C")

        self.btn_generate = QPushButton("Generate Profile")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        # Fan toggle button
        self.btn_fans = QPushButton("Fans: OFF")
        self.btn_fans.setMinimumHeight(40)

        # BIG emergency off button
        self.btn_emergency_off = QPushButton("⛔ EMERGENCY HEATER OFF")
        self.btn_emergency_off.setMinimumHeight(70)
        self.btn_emergency_off.setStyleSheet("""
            QPushButton {
                font-size: 22px;
                font-weight: 800;
                background-color: #b00020;
                color: white;
                border-radius: 12px;
                padding: 12px;
            }
            QPushButton:pressed { background-color: #7a0016; }
        """)

        self.lbl_status = QLabel(
            f"Status: idle | heaterGPIO={HEATER_GPIO_BCM} | fansGPIO={FANS_GPIO_BCM} | loop={CONTROL_LOOP_HZ}Hz | PWM={HEATER_PWM_HZ}Hz | PV=sensor{CONTROL_SENSOR_INDEX}"
            f" | cal={CALIBRATION_CSV_PATH} | LPF tau={TEMP_LP_TAU_S}s"
        )
        self.lbl_now = QLabel("stage=-- | t=-- min | SP=-- °C | PV=-- °C | avg=-- °C | T0/T1/T2=--/--/-- | heater=-- | duty=--")

        # Plot
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time (min)")
        self.plot.setLabel("left", "Temperature (°C)")
        self.plot.addLegend()

        # Curves (thickness only)
        self.curve_sp = self.plot.plot([], [], name="Setpoint", pen=pg.mkPen(width=3))
        self.curve_avg = self.plot.plot([], [], name="Average", pen=pg.mkPen(width=2))

        self.curve_t0 = self.plot.plot([], [], name="Temp sensor0 (cal+LPF)", pen=pg.mkPen(width=1))
        self.curve_t1 = self.plot.plot([], [], name="Temp sensor1 (cal+LPF)", pen=pg.mkPen(width=1))
        self.curve_t2 = self.plot.plot([], [], name="Temp sensor2 (cal+LPF)", pen=pg.mkPen(width=1))

        self.curve_pv = self.plot.plot([], [], name=f"PV (sensor{CONTROL_SENSOR_INDEX})", pen=pg.mkPen(width=1))

        # Data buffers (history only up to "now")
        self.profile: Profile | None = None
        self.hist_t_min: list[float] = []
        self.hist_sp: list[float] = []
        self.hist_avg: list[float] = []
        self.hist_pv: list[float] = []
        self.hist_t0: list[float] = []
        self.hist_t1: list[float] = []
        self.hist_t2: list[float] = []

        # Layout
        controls = QGroupBox("Profile Settings")
        c_layout = QVBoxLayout()

        def row(label: str, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget)
            c_layout.addLayout(h)

        row("Initial temp (preheat):", self.initial_temp)
        row("Ramp rate:", self.ramp)
        row("Hold temp:", self.hold_temp)
        row("Hold time:", self.hold_time)
        row("Soak rate:", self.soak)
        row("End temp:", self.end_temp)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_generate)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        c_layout.addLayout(btns)

        controls.setLayout(c_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.btn_emergency_off)
        layout.addWidget(self.btn_fans)
        layout.addWidget(controls)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_now)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        # Wiring
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_emergency_off.clicked.connect(self.on_emergency_off)
        self.btn_fans.clicked.connect(self.on_toggle_fans)

        self.worker: ControlThread | None = None

    def _set_fans_ui(self, on: bool):
        self.fans_on = bool(on)
        if self.fans_on:
            self.btn_fans.setText("Fans: ON")
            self.btn_fans.setStyleSheet("font-weight: 700;")
        else:
            self.btn_fans.setText("Fans: OFF")
            self.btn_fans.setStyleSheet("")

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

    def _clear_history(self):
        self.hist_t_min.clear()
        self.hist_sp.clear()
        self.hist_avg.clear()
        self.hist_pv.clear()
        self.hist_t0.clear()
        self.hist_t1.clear()
        self.hist_t2.clear()

        self.curve_sp.setData([], [])
        self.curve_avg.setData([], [])
        self.curve_pv.setData([], [])
        self.curve_t0.setData([], [])
        self.curve_t1.setData([], [])
        self.curve_t2.setData([], [])

    def on_emergency_off(self):
        if self.worker is not None:
            self.worker.force_heater_off()
        self.lbl_status.setText("Status: EMERGENCY OFF pressed — heater forced OFF")

    def on_generate(self):
        try:
            start_for_profile = float(self.initial_temp.value())

            self.profile = build_profile(
                start_temp_c=start_for_profile,
                ramp_c_per_min=float(self.ramp.value()),
                hold_temp_c=float(self.hold_temp.value()),
                hold_time_min=float(self.hold_time.value()),
                soak_c_per_min=float(self.soak.value()),
                end_temp_c=float(self.end_temp.value()),
            )

            self._clear_history()

            self.lbl_status.setText(
                f"Status: profile generated (preheat to {self.initial_temp.value():.1f} °C, main length {self.profile.total_s}s)"
            )

        except Exception as e:
            QMessageBox.critical(self, "Generate profile failed", str(e))

    def on_start(self):
        if self.profile is None:
            QMessageBox.warning(self, "No profile", "Generate the profile first.")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_generate.setEnabled(False)

        self._clear_history()

        initial_target = float(self.initial_temp.value())

        try:
            self.worker = ControlThread(profile=self.profile, initial_target_c=initial_target)
        except Exception as e:
            QMessageBox.critical(self, "Start failed", str(e))
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_generate.setEnabled(True)
            return

        self.worker.update.connect(self.on_update)
        self.worker.fault.connect(self.on_fault)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

        self.lbl_status.setText("Status: running (reading Pico + calibrated+LPF temps + PWM heater)")

    def on_stop(self):
        if self.worker is not None:
            self.worker.stop()
        self.lbl_status.setText("Status: stopping...")

    def on_update(self, d: dict):
        stage = d.get("stage", "--")
        sp = float(d.get("setpoint_c", float("nan")))
        pv = float(d.get("actual_c", float("nan")))
        temps = d.get("temps_c", [float("nan"), float("nan"), float("nan")])
        t0, t1, t2 = (float(temps[0]), float(temps[1]), float(temps[2]))
        forced = bool(d.get("forced_off", False))
        duty = float(d.get("duty", 0.0))

        # time axis in minutes, using total runtime so the plot is continuous across stages
        t_total_s = int(d.get("t_total_s", 0))
        t_min = t_total_s / 60.0

        # average (ignore NaNs)
        vals = [t0, t1, t2]
        good = [x for x in vals if not math.isnan(x)]
        avg = sum(good) / len(good) if good else float("nan")

        # append history
        self.hist_t_min.append(t_min)
        self.hist_sp.append(sp)
        self.hist_avg.append(avg)
        self.hist_pv.append(pv)
        self.hist_t0.append(t0)
        self.hist_t1.append(t1)
        self.hist_t2.append(t2)

        # plot (history only)
        self.curve_sp.setData(self.hist_t_min, self.hist_sp)
        self.curve_avg.setData(self.hist_t_min, self.hist_avg)

        self.curve_pv.setData(self.hist_t_min, self.hist_pv)
        self.curve_t0.setData(self.hist_t_min, self.hist_t0)
        self.curve_t1.setData(self.hist_t_min, self.hist_t1)
        self.curve_t2.setData(self.hist_t_min, self.hist_t2)

        heater_txt = "FORCED OFF" if forced else ("PWM")
        note = d.get("note", "")
        note_txt = f" | {note}" if note else ""

        self.lbl_now.setText(
            f"stage={stage} | t={t_min:>6.2f} min | SP={sp:>6.1f} °C | PV={pv:>6.2f} °C | avg={avg:>6.2f} °C"
            f" | T0/T1/T2={t0:>6.2f}/{t1:>6.2f}/{t2:>6.2f} | heater={heater_txt} | duty={duty:0.2f}{note_txt}"
        )

        if stage == "initial" and (not math.isnan(pv)) and pv >= (sp - INITIAL_BAND_C):
            self.lbl_status.setText("Status: preheat reached — starting main cycle...")
        if d.get("done", False):
            self.lbl_status.setText("Status: profile complete (heater OFF)")

    def on_fault(self, msg: str):
        QMessageBox.critical(self, "Fault", msg)
        self.lbl_status.setText(f"Status: fault — {msg}")
        self.on_stop()

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_generate.setEnabled(True)
        self.worker = None

    def closeEvent(self, event):
        # Ensure fans OFF on exit
        try:
            self.fans.off()
        except Exception:
            pass

        if self.worker is not None:
            self.worker.force_heater_off()
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 650)
    w.show()
    sys.exit(app.exec())
