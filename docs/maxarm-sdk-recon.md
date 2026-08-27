> **Resolution note (2026-08-25).** This report was produced from the public Hiwonder docs before the
> my earlier project was found. The open questions in section (h) are resolved by the firmware
> source (`MaxArm_Serial_Communication.zip`) and by `~/Documents/connect 4 bot/maxarm.py`, which was
> validated on this arm on 2026-08-17: the `MaxArm_micropython_microUSB` slave firmware **is installed**
> (factory files backed up in that project); it opens `UART(1, 9600, tx=1, rx=3)` = the micro-USB port,
> so no USB-TTL adapter is needed; **9600 baud**; a plain port open does **not** reset the board (a
> DTR/RTS pulse does); after a reset the firmware sleeps 10 s then homes (~3 s) before answering;
> reply checksums cover the whole frame including `AA 55`; servo read-back matches to ~8 mm; macOS port
> is `/dev/cu.usbserial-*` (the number changes with the USB socket). The doc's `AA 55 07 01 02 f6`
> suction example is simply a typo (rule gives `f5`); the docs' `AA 55 03 08 78 00 4c ff 55 00 e8 03 f1`
> example is correct and is a unit-test vector in `tests/test_arm_protocol.py`.

# Hiwonder MaxArm — SDK Recon Report

Sources: Hiwonder MaxArm v1.0 docs saved as text in `scratchpad/maxarm_docs/` (files `4.*`, `5.*`, `6.*`, `8.*`, `10.*`; cited below as `(§N, lines)`), each line-cited claim re-grepped in the file. Web sources are cited by URL. Where a doc extractor and the grep verifier disagreed, the verifier wins (noted inline).

Citation key:
- §4 = `4.Underlying_Program_Learning_checked.txt`
- §5 = `5.Hardware_Basic_Learnig_formatted.txt`
- §6 = `6.Secondary_Development_checked.txt`
- §8 = `8.Inverse_Kinematics_Basic_andApplication_formatted.txt`
- §10 = `10.MaxArm_Serial_Communication_formatted.txt`

---

## TL;DR

- **Every Python example in §4/§5/§6/§8 is MicroPython running ON the ESP32**, not on the laptop. The laptop is only used to download `main.py` to the board and press reset (§4 lines 262-274; §6 3619-3623; §8 877-883).
- **The laptop-side path is §10 "MaxArm Serial Communication"**: MaxArm is the *slave*; the PC is the *master* sending binary frames `AA 55 | func | len | data | checksum` at **9600 8N1** (§10 lines 75, 117-129, 191-207).
- Move = `FUNC_SET_XYZ` 0x03, payload `<hhhH` (int16 x, y, z; uint16 time ms). Suction = `FUNC_SET_SUCTIONNOZZLE` 0x07, one byte: 1 = pump on, 2 = pump off + valve open, 3 = valve close (§10 2829-2851).
- **No acknowledgement for any SET command**; completion is inferred from the commanded duration, or by polling `FUNC_READ_XYZ` 0x13 (§10 2831-2871; 2412-2419).
- **This protocol only works if the "MaxArm_micropython_microUSB" (or Arduino) slave firmware is on the board.** The factory firmware does not speak it (web hunt: `https://raw.githubusercontent.com/aarnavshah12/connect4-robot/HEAD/maxarm_factory_backup/main.py`). Owner must confirm which firmware is loaded.

---

## (a) How the laptop talks to the arm

### Master/slave roles
> "Throughout this chapter, MaxArm serves as a slave device, communicating with other devices via UART serial communication for data transmission." (§10, line 75)

The master can be a PC, Raspberry Pi, Arduino, or STM32 (§10, line 73).

### Where does Python run?
**On the board, in chapters 4/5/6/8.** The docs' Python workflow is: open the Hiwonder "Python editor", download `.py` files to the "device", then press reset:
> "① Open the program ADC/main.py in Workspace. Click on the download icon in menu bar, or directly right click and select "download" the program file." (§4, 264)
> "③ After downloading, the program is saved in the list of "device"." (§4, 268)
> "④ Click on the reset icon or press the reset button on ESP32 controller." (§4, 274)

The on-board code uses MicroPython-only APIs (`from machine import Pin,ADC`, `time.sleep_ms`) (§5 3306-3307, 3044; §8 937). The REPL note confirms it: "UART0 is occupied by REPL and cannot be used" (§5, 3772). The docs link to `docs.micropython.org` (§4, 356).

The firmware files that live ON the ESP32 (§4, 364-377 and 404-438): `boot.py`, `main.py`, `BusServo.py`, `Key.py`, `Buzzer.py`, `Led.py`, `PWMServo.py`, `RobotControl.py`, `USBDevice.py` ("Controller USB driver program: parse serial data", §4 428-429), `SuctionNozzle.py`, `__espmax.mpy` ("Kinematics base library", §4 434-435), `espmax.py` ("Kinematic package library: call kinematics base library to get solution and drive servo", §4 437-438). All grep-verified.

**Consequence for a laptop controller:** you cannot `import espmax` on the laptop. A laptop-side `arm.py` must speak the §10 serial protocol to a slave firmware running on the ESP32, which internally calls `self.arm.set_position(...)` / `self.nozzle.on_uart()` etc. (§10, 1147-1153, 1196-1201).

### Physical link and firmware variants
§10 documents two slave interfaces, selected by which firmware is flashed / which port argument `begin()` gets:
> "MaxArm's serial communication supports two interfaces, one is the 4-pin interface, suitable for devices with pin interfaces for UART communication ... The other interface is the micro-USB interface, suitable for devices with USB master interfaces, such as Raspberry Pi, Jetson Nano and others." (§10, 175)

MicroPython slave firmware (`MaxArm_ctl.py` on the board), §10 lines 878-887 (verbatim):
```python
  def begin(self , port):
    if port == SELECT_PORT.PORT_FOR_USB:
      print("begin in USB")
      self.__uart = UART(1, 9600, tx=10, rx=9)
    else:
      print("begin in 4Pin")
      self.__uart = UART(1, 9600, tx=33, rx=32)
    self.nozzle = SuctionNozzle()
    self.arm.go_home(1500)
    time.sleep_ms(2000)
```
`SELECT_PORT.PORT_FOR_USB = const(0x01)`, `PORT_FOR_4Pin = const(0x03)` (§10, 901-903).

Note: the shipped zip (`https://docs.hiwonder.com/projects/MaxArm/en/latest/_static/source_code/MaxArm_Serial_Communication.zip`, file `MaxArm_micropython_microUSB/MaxArm_ctl.py`) reportedly has `UART(1 , 9600 , tx=1, rx=3 )` for the USB branch, i.e. the USB-REPL pins — this differs from the doc page's `tx=10, rx=9`. **Unverified against the zip; check the file actually on the board.**

The §10 PC wiring section uses a USB-TTL adapter on the 4-pin header, not the micro-USB port: "Connect the RXD, TXD, GND of USB adapter to IO32, IO33, GND ports of ESP32 expansion board respectively using Dupont wires." (§10, 111). Ground must be shared and TX/RX crossed (§10, 157-159).

### Baud rate and format
> Baud rate 9600 / Data bits 8 / Parity bit None / Stop bit 1 (§10, 117-129; grep confirms `9600` at line 120)

Master must use 9600: "initialize the serial baud rate to 9600, otherwise communication cannot be established normally." (§10, 2361). The official Python master constructor's *default* is `baudrate=115200` (§10, 2064) but every documented call overrides it: `ma = MaxArm_ctl.MaxArm_ctl(device = "/dev/ttyUSB0", baudrate=9600)` (§10, 2369 and 2505). The `115200` values elsewhere are the internal bus-servo UART2 (§5 382-385, pins RX 35 / TX 12, §5 360-361) and Arduino debug `Serial.begin(115200)` (§10, 286) — not the PC link.

### Port naming
- Linux: `/dev/ttyUSB0` (§10, 2369).
- Windows: `COMx`, identify by "CH340" in Device Manager; never `COM1` (§4 176, 210; §10 2721).
- macOS: **not documented anywhere in the doc set.** The CH340 chip implies `/dev/cu.usbserial-*`/`/dev/cu.wchusbserial*`; my earlier driver used `/dev/cu.usbserial-310` (`https://raw.githubusercontent.com/aarnavshah12/connect4-robot/HEAD/maxarm.py`). Inference, not doc-stated.

### Boot behaviour (affects when the first command can be sent)
MicroPython slave: on `begin()` it homes over 1500 ms then sleeps 2000 ms (§10, 886-887). The zip's `main.py` reportedly sleeps 10 s first ("Wait for 10s to prevent program download failure") then polls `rec_data()` every 100 ms (web hunt, zip `MaxArm_micropython_microUSB/main.py`; not in the doc text files — verify on the board). Factory `main.py` starts Bluetooth taking ~10 s (§4, 385).

---

## (b) Exact API for moving the end effector to (x, y, z)

### Laptop side (serial protocol, §10)
**Command:** `FUNC_SET_XYZ`, function code `0x03`, data length 8 (§10, 219-221, 2829).
> "Data Information: First are the coordinates corresponding to XYZ, followed by the operating time, totaling 4 values. Each value is split into 2 bytes, with the low byte first and the high byte last." (§10, 2833)
> Example: `AA 55 03 08 78 00 4c ff 55 00 e8 03 f1` → X=120, Y=-180, Z=85, time=1000 ms (§10, 2835)

Official Python master method (`MaxArm_ctl.py`, Raspberry Pi routine), §10 lines 2152-2168 (verbatim):
```python
    def set_xyz(self , pos , time):        
        # Pack position data and time into a byte string (将位置数据和时间打包为字节串)
        pos_bytes = struct.pack('<hhh', *pos)
        time_bytes = struct.pack('<H', time)

        # Build the data frame to be sent (构建要发送的数据帧)
        msg = bytearray([0xAA, 0x55, PACKET_FUNCTION.FUNC_SET_XYZ, 0x08])
        msg.extend(pos_bytes)
        msg.extend(time_bytes)

        # Calculate checksum (计算校验位)
        checksum = checksum_crc8(msg[2:]) # Calculate checksum (计算校验位)

        msg.append(checksum)

        # Send data frame (发送数据帧)
        self.serial_send(msg)
```
- Signature: `set_xyz(self, pos, time)` (verifier: method of class `MaxArm_ctl`, line 2152; the extractor's name `MaxArm_ctl.set_xyz` is not verbatim).
- `pos`: 3 values packed as signed int16 little-endian; "a list of length 3" (§10, 2412). Units are **never stated as mm on the §10 page**; mm comes from §5/§6/§8 (see (e)).
- `time`: uint16, "the duration of operation (in milliseconds)" (§10, 2412).
- Return value: none (falls off after `serial_send`). No reply from the arm.
- Caveat: the doc example passes a Python **set** `xyz = {120 , -180 , 85}` (§10, 2416) — use a list/tuple.

### On the board (what the frame invokes)
Slave handler, §10 lines 1147-1153 (verbatim):
```python
    elif ctl_com.function == PACKET_FUNCTION.FUNC_SET_XYZ:
      # print("FUNC_SET_XYZ")
      if len == 8:
        xyz = bytes(ctl_com.data[:7])
        time_count = (ctl_com.data[6] & 0x00FF) | ((ctl_com.data[7] << 8 ) & 0xFF00)
        unpacked_data = struct.unpack('<hhh', xyz) 
        self.arm.set_position(unpacked_data , time_count)
```
On-board MicroPython API (`espmax.ESPMax`): `arm.set_position((x, y, z), duration_ms)`:
> "Use the arm.set_position() function to control robotic arm. Take the code arm.set_position((0,-160,85),1500) as example. / The first parameter "(0,-160,85)" is the position of the suction nozzle on x, y and z axes. / The second parameter "1500" represents the running time and the unit is ms." (§6, 3993-3997)

Construction: `bus_servo = BusServo()` / `arm = ESPMax(bus_servo)` (§8, 1020-1021; §6, 4040-4041). `arm.go_home()` with no argument in all Python listings (§8, 1034; §6, 3829). `(x,y,z) = arm.ORIGIN` reads the home coordinates (§8, 910).

The Python `espmax.py` source is **not printed in any doc page** — §5 and §6 paste the C++ `ESPMax.cpp` under the "Python" heading (§5, 339; §6, 3387). The C++ equivalent, §6 lines 453-469:
```c
int set_position(float pos[3], int duration){
    float x = pos[0];
    float y = pos[1];
    float z = pos[2];
    if(z > 255) z = 255;
    if(sqrt(x*x + y*y) < 50) return int(0);
    float angles[3];
    inverse(pos,angles);
    float pul[3];
    deg_to_pulse(angles,pul);
    for(int i=0; i<3; i++){
        positions[i] = pul[i];
        BusServo.LobotSerialServoMove(i+1,pul[i],duration);
        delay(2);
    }
    return int(1);
}
```
Return: 1 accepted, 0 rejected (radius < 50 mm). The web hunt reports the MicroPython `espmax.py` in the zip clamps `z > 225` and returns `None`/`False` on rejection (zip URL above) — consistent with the prose "The coordinate of z-axis can not be greater than 225" (§6, 473) but not with the C++ `255` (§6, 457). **The rejection is silent over serial** — nothing is sent back to the master.

---

## (c) Exact API for suction pump / valve

### Laptop side
**Command:** `FUNC_SET_SUCTIONNOZZLE`, function code `0x07`, data length 1 (§10, 227-229, 2845).
> "Data Information: 1 indicates turning on the vacuum pump, 2 indicates turning off the vacuum pump and opening the air valve, 3 indicates closing the air valve." (§10, 2849)
> Example: `AA 55 07 01 02 f6` (sub-command 2) (§10, 2851)

Official Python master, §10 lines 2228-2236 (verbatim; the capital-N typo is real, verifier line 2228):
```python
    def set_SuctioNnozzle(self , func):
        if func not in [1,2,3]:
            return
        # Build the data frame to be sent (构建要发送的数据帧)
        data = bytearray([0xAA, 0x55, PACKET_FUNCTION.FUNC_SET_SUCTIONNOZZLE, 0x01, func & 0xFF])

        crc = checksum_crc8(data[2:])
        data.append(crc)
        self.serial_send(data)
```
Semantics and required release sequence:
> "The value involves 1, 2 and 3 corresponding to opening the air pump, closing the air pump and opening the air valve, and closing the air value, respectively." (§10, 2211)
> "when the pump is opened, the suction cup will generate suction force. Then, when the pump is closed and the valve is opened, negative pressure still exists inside the suction cup after the pump is turned off. Opening the valve allows external air to enter, eliminating the negative pressure and completely eliminating the suction force. Finally, the valve is closed." (§10, 2213)

Documented example (§10, 2382-2386): `set_SuctioNnozzle(1)`, sleep 2, `set_SuctioNnozzle(2)`, sleep 0.2, `set_SuctioNnozzle(3)`.

Full frames computed by me from the documented rule `(~(0x07+0x01+cmd)) & 0xFF` (only sub-command 2 is printed in the doc): cmd 1 → `AA 55 07 01 01 F6`, cmd 2 → `AA 55 07 01 02 F5`, cmd 3 → `AA 55 07 01 03 F4`. The doc prints `AA 55 07 01 02 f6` for cmd 2 (§10, 2851), which does **not** satisfy the rule (`~0x0A & 0xFF = 0xF5`). The other printed examples do satisfy it (SET_ANGLE → `6d`, SET_XYZ → `f1`, SET_PWMSERVO → `34`, READ_XYZ request → `EC`; §10 2827, 2835, 2843, 2865 — all recomputed by me). So the suction example on the page is most likely a typo; compute checksums with the code rule (`checksum_crc8(data[2:])`, §10 2234) rather than copying hex from the page.

### On the board
Slave dispatch, §10 lines 1196-1201 (verbatim):
```python
        if ctl_com.data[0] == 1:
          self.nozzle.on_uart()
        elif ctl_com.data[0] == 2:
          self.nozzle.off_uart_1()
        elif ctl_com.data[0] == 3:
          self.nozzle.off_uart_2()
```
(Prose at §10 1177-1181 calls them `on_uart()`, `on_uart1()`, `on_uart2()` — the code names above are authoritative.) Arduino slave maps 1→`Pump_on()`, 2→`Valve_on()`, 3→`Valve_off()` (§10, 599-607).

Non-serial on-board API (for reference): `nozzle = SuctionNozzle()`; `nozzle.on()  # Turn on the pump, close the solenoid valve at the same time`; `nozzle.off()  # Turn off the pump, open the solenoid valve at the same time` (§5, 3701-3707); "use the nozzle.on() function to turn on air pump. If it is nozzle.off() function, air pump will be off." (§6, 5401). Hardware: "Air pump is connected to M1 port and the solenoid valve to M2 port" (§5, 1947). Nozzle rotation is a separate PWM servo: `nozzle.set_angle(30,600)` — angle in degrees, time in ms (§6, 3839-3843); over serial it is `FUNC_SET_PWMSERVO` 0x05, pulse 500-2500 + time (§10, 2837-2843).

---

## (d) Completion signalling

**There is none for SET commands.** Facts:
- Frame with bad checksum or unknown function is dropped silently: "If correct, the corresponding function is called; otherwise, the data frame is skipped." (§10, 205)
- Only `FUNC_READ_ANGLE` (0x11) and `FUNC_READ_XYZ` (0x13) produce a reply (§10, 2857, 2871). `set_xyz` ends with `self.serial_send(msg)` and returns nothing (§10, 2168).
- On the board, `set_position` fires timed servo moves and returns immediately (`LobotSerialServoMove(i+1,pul[i],duration); delay(2);` then `return int(1);`, §6 465-468). `duration` is "the time for the nozzle to move to the target position and the unit is ms" (§6, 4685); "t: total movement time (the longer the time, the slower the speed)" (§6, 224).
- Every doc example sleeps after a move: `ma.set_xyz(xyz , 1000)` / `time.sleep(2)` (§10, 2417-2418); `arm.set_position((x,y,z-100),2000)` / `time.sleep_ms(2000)` (§8, 936-937); `arm.set_position((0,-120,80),1500)` / `time.sleep(1.6)` (§8, 1060-1061).
- Feedback available: `FUNC_READ_XYZ` request `AA 55 13 00 EC`, reply `AA 55 13 06 <int16 x><int16 y><int16 z> <chk>` (11 bytes) (§10, 2865-2871). On board it is forward kinematics from real servo positions: `(x, y, z) = self.arm.read_position()` (§10, 1240); Arduino `read_position` polls `LobotSerialServoReadPosition(i+1)` (§6, 499-505). Official master `read_xyz` sends, `time.sleep(0.1)`, then `self.__uart.read(11)` (§10, 2326-2347).

Reply-checksum caveat (from firmware code, §10 1241-1245): the MicroPython slave computes `checksum_crc8(0,0,send_data)` over the whole `send_data` including `AA 55`, whereas the master's request checksum covers `msg[2:]`. Two hardware-tested third-party drivers report the reply checksum therefore includes the header and is off-by-one vs the request rule (`https://github.com/Nu424/maxarm-python`; `https://raw.githubusercontent.com/aarnavshah12/connect4-robot/HEAD/maxarm.py`). Do not reuse the official `rec_handle` verbatim; also official `read_angles` reads 12 bytes of an 11-byte reply with no timeout (§10, 2276) and would block.

---

## (e) Coordinate frame, units, home, limits, example coordinates

- Units/origin: "MaxArm uses x-y-z axes coordinate system (unit:mm) and takes the the base centre of robotic arm as original point (0,0,0)" (§5, 123; same at §6, 103). Over serial, coordinates are int16 (1-unit resolution) and §10 never says "mm" (§10, 2833).
- Home / `ORIGIN`: `float ORIGIN[3] ={ 0, -(L1 + L3 + L4), (L0 + L2)};` (§6, 359) with `L0 84.4 / L1 8.14 / L2 128.4 / L3 138.0 / L4 16.8` (§6, 265-269) → **(0, -162.94, 212.8)**. Prose prints "y=162.94" without the sign (§5, 291) — code is authoritative. The zip's MicroPython `espmax.py` reportedly uses `L0=84.0, L1=8.2, L2=128.0, L3=138.0, L4=16.8` → (0, -163, 212) (web hunt, zip URL) — unverified by me. Python reads it via `(x,y,z) = arm.ORIGIN` (§8, 910).
- Home is at the workspace edge: "The initial position is already at the edge of the movable workspace, so move down first; otherwise, the arm cannot move along X and Y axes" (§5, 242; §6, 221-222; §8, 282).
- Axis directions — **the docs contradict themselves**:
  - Table (§5, 130-137; §6, 107-118): +x right, +y forward, **+z down**.
  - Code comments: `z-100` = "Move Z axis down 100 mm" (§8, 936), `x-50` = "Move X axis left 50 mm", `x+50` = "right", `y-50` = "backward", `y+50` = "forward" (§8, 941-951); pick at z=85, "Lift up" at z=200 (§6, 3799-3803). So **+z is up in all code**; the table's z sentence is wrong.
  - Prose "to move 200mm to the left ... set x value plus 200. If want to move to 200 to the right, set x-200." (§5, 331) contradicts the table on x.
  - All working examples pick/place at **negative y** (y = -120 … -280), so the front of the arm is -y (§6, 3797-3811; §8, 1060-1076; §10, 2416-2419).
- Limits (code): `if(z > 255) z = 255;` and `if(sqrt(x*x + y*y) < 50) return int(0);` (§6, 457-458); prose says z max 225 (§6, 473). Servo pulse limits: ID3 ≥ 470, ID2 ≤ 700 (§6, 383-384). Bus servo: pulse 0-1000 = 0-240° (§5, 1165). No max-reach number anywhere in the docs.
- Example coordinates seen in docs (mm): (0,-160,100) approach, (0,-160,85) pick, (0,-160,200) lift, (70,-150,90), (130,-150,88), (160,0,200), (160,0,88+40n) stacking, (150,-35,90), (150,10,88), (70,-165,86), (120,-140,85), (120,-80,85), (120,-20,82) (§6, 3797-3811, 3941-3943, 4569-4571, 5087-5097); (0,-120,80), (0,-280,75), (0,-280,150), (100,-200,150), (100,-200,80), (50,-260,80), x∈[-100,100] at y=-200 (§8, 1060-1076, 1182-1193); serial examples (120,-180,85), (-120,-180,85), read-back (-159,-6,96) (§10, 2416-2419, 2871).

---

## (f) Serial protocol packet format (verbatim, §10)

Frame (§10, 191-207): `0xAA 0x55 | func | len | data | check`
> "Frame header: if 0xAA and 0x55 are received sequentially, it indicates that there is data to be received, consisting of a fixed 2 bytes. / Function Code: Used to indicate the purpose of an information frame, consists of 1 byte. / Data Length: Indicates the number of data bits carried by the data frame. / Check Bit: ... The calculation method for the check bit is: calculate the sum of the function code, data length, and data, then take the complement, and finally, take the low byte, which serves as the checksum."

Checksum (C reference, §10 1374-1383, verbatim):
```c
static uint16_t checksum_crc8(const uint8_t *buf, uint16_t len)
{
    uint8_t check = 0;
    while (len--) {
        check = check + (*buf++);
    }
    check = ~check;
    return ((uint16_t) check) & 0x00FF;
}
```
Header constants: `CONST_STARTBYTE1` = 0xAA, `CONST_STARTBYTE2` = 0x55 (§10, 1389; verifier: each exists individually).

Function-code table (§10, 215-237, verbatim names):

| Name | Code | len | Payload (all little-endian) | Doc example |
|---|---|---|---|---|
| `FUNC_SET_ANGLE` | 0x01 | 8 | 3× uint16 servo pos (0-1000) + uint16 time ms | `AA 55 01 08 c8 00 f4 01 f4 01 d0 07 6d` (§10, 2827) |
| `FUNC_SET_XYZ` | 0x03 | 8 | int16 x, y, z + uint16 time ms | `AA 55 03 08 78 00 4c ff 55 00 e8 03 f1` (§10, 2835) |
| `FUNC_SET_PWMSERVO` | 0x05 | 4 | uint16 pulse (500-2500) + uint16 time ms | `AA 55 05 04 D0 07 e8 03 34` (§10, 2843) |
| `FUNC_SET_SUCTIONNOZZLE` | 0x07 | 1 | 1 pump on / 2 pump off+valve open / 3 valve close | `AA 55 07 01 02 f6` (§10, 2851; see checksum note in (c)) |
| `FUNC_READ_ANGLE` | 0x11 | 0 | — ; reply `AA 55 11 06 <3×int16> chk` | req `AA 55 11 00 EE`, reply `AA 55 11 06 60 03 9A 01 C9 02 20` (§10, 2857-2859) |
| `FUNC_READ_XYZ` | 0x13 | 0 | — ; reply `AA 55 13 06 <x><y><z> chk` | req `AA 55 13 00 EC`, reply `AA 55 13 06 61 FF FA FF 60 00 2E` (§10, 2865-2871) |

Slave reply construction (§10, 1238-1245): `send_data = bytearray([0xAA,0x55,0x13,0x06]); send_data += struct.pack('<hhh', x, y, z); check_num = checksum_crc8(0,0,send_data); send_data.append(check_num)`.

---

## (g) Key code listings (verbatim, with source lines)

**1. Official PC/RPi master class `MaxArm_ctl` — constructor and read_xyz (§10, 2064-2071, 2326-2347)**
```python
    def __init__(self, device = "/dev/ttyUSB0", baudrate=115200):
        self.__uart = serial.Serial(
                                        port=device,
                                        baudrate=baudrate,
                                        bytesize=serial.EIGHTBITS,
                                        parity=serial.PARITY_NONE,
                                        stopbits=serial.STOPBITS_ONE,
                                    )
```
```python
    def read_xyz(self):
        # Build the command frame for reading xyz coordinates (构建要发送的读取角度的命令帧)
        command = bytearray([0xAA, 0x55, 0x13, 0x00 , 0xEC])
        # Send the command frame to read xyz (发送读取角度的命令帧)
        self.serial_send(command)

        time.sleep(0.1)

        # Receive data from serial (从串口接收数据)
        response = self.__uart.read(11)

#        print(response)

        # Call the parsing function to process received data (调用解析函数处理接收到的数据)
        rec = self.rec_handle(response , 0x13)

        if rec:
            print(len(rec))
            xyz = struct.unpack('<hhh', rec)
        else:
            return
        return xyz
```
(`serial_send`, `map_func`, `rec_handle`, `checksum_crc8` Python bodies and `PACKET_FUNCTION` are referenced but **not printed** on the page — verifier lines 2125, 2106, 2281, 2121, 983-988.)

**2. Official master usage example `main.py` (§10, 2369, 2380-2419)** — see (b)/(c); key lines:
```python
ma = MaxArm_ctl.MaxArm_ctl(device = "/dev/ttyUSB0", baudrate=9600)
    ma.set_SuctioNnozzle(1) #Turn on the pump (打开气泵)
    time.sleep(2)
    ma.set_SuctioNnozzle(2) #Open solenoid valve and turn off pump (打开电磁阀并关闭气泵)
    time.sleep(0.2)
    ma.set_SuctioNnozzle(3) #Close solenoid valve (关闭电磁阀)
    xyz = {120 , -180 , 85} # Set xyz coordinates to 120/-180/85, with run time of 1000ms (将xyz坐标设置为120/-180/85,运行时间为1000ms)
    ma.set_xyz(xyz , 1000)
    time.sleep(2)
```

**3. MicroPython slave `begin()` and `deal_command()` FUNC_SET_XYZ / SUCTION branches** — §10, 878-887, 1147-1153, 1196-1201; printed verbatim in (a), (b), (c) above.

**4. On-board pick/place pattern (§6, 3796-3817, MicroPython)**
```python
      arm.set_position((0,-160,100),1500)
      time.sleep_ms(1000) #Wait for 1000ms 
      arm.set_position((0,-160,85),800) #Suction the color block 
      nozzle.on()  #Turn on the pump 
      time.sleep_ms(1000)
      arm.set_position((0,-160,200),1000) #Lift up 
      time.sleep_ms(1000)
      ...
      nozzle.off()  #Turn off suction pump 
      arm.set_position((130,-150,200),1000) #Lift up 
      time.sleep_ms(1000)
      arm.go_home() #Reset arm to initial position 
```

**5. On-board IK demo (§8, 899-953 fragments)**
```python
import time
from espmax import ESPMax
from BusServo import BusServo
  (x,y,z) = arm.ORIGIN  # Read the initial XYZ position of the arm (读取机械臂初始位置的XYZ坐标)
  arm.set_position((x,y,z-100),2000) # Move Z axis down 100 mm relative to initial position (Z轴相对初始位置下移100毫米)
  time.sleep_ms(2000)
```

**6. C++ `ESPMax.cpp` `set_position`/`go_home`/`read_position` and `_espmax.h` link constants** — §6, 265-269, 359, 453-469, 483-485, 499-505; `set_position` printed in (b).

---

## (h) Open questions to confirm on the physical kit

1. **Which firmware is on the ESP32?** Factory firmware (BLE + handle receiver) does not implement the `AA 55` protocol; the `MaxArm_micropython_microUSB` slave files from the §10 zip must be loaded (web: connect4-robot `PROGRESS.md` / `maxarm_factory_backup/main.py`). Confirm by sending `AA 55 13 00 EC` at 9600 and checking for an 11-byte `AA 55 13 06 …` reply.
2. **Which UART pins the loaded slave firmware uses** (`tx=10, rx=9` per doc page §10 881 vs `tx=1, rx=3` reportedly in the zip). If it is the micro-USB variant, plain USB works; if it is the 4-pin variant you need a USB-TTL adapter on IO32/IO33 (§10, 111).
3. **macOS port name** — not documented; expect a CH340 `/dev/cu.usbserial-*` (my previous rig: `/dev/cu.usbserial-310`).
4. **Does opening the port reset the board?** Conflicting third-party reports; both agree there is a 10 s + ~3.5 s silent window after reset (web sources). Verify empirically how long until the first `read_xyz` answers.
5. **Suction sub-command checksums** — the doc's `AA 55 07 01 02 f6` disagrees with the documented rule (which gives `F5`). Verify which byte the board accepts.
6. **Reply checksum convention** (whole frame incl. header vs func/len/data) — verify one `read_xyz` reply by hand.
7. **x-axis sign** (table says +x right, prose says +x left) and **y sign** — teach two points and read them back. z: confirm +z up (all code agrees).
8. **z clamp** 225 vs 255 and whether the MicroPython firmware silently drops out-of-reach targets — verify with a `read_xyz` after a deliberately unreachable command.
9. **`go_home()` default duration** in the MicroPython `espmax.py` (not printed in docs; zip reportedly `duration=2000`).
10. **Servo-angle mapping** for `FUNC_SET_ANGLE` (Python master maps 0-180→0-1000, §10 2106; prose says 0-240°, §10 2391) — irrelevant if only XYZ is used.
11. **Actual reachable workspace** for the defect-detect rig — the docs give no max radius; my previously taught points (`https://raw.githubusercontent.com/aarnavshah12/connect4-robot/HEAD/poses.json`) are the best evidence.

---

## (i) Recommended integration approach for a laptop-side `arm.py`

1. **Depend only on `pyserial`.** Open the CH340 port at **9600 8N1** with a read timeout (e.g. 1 s) — the official class has none (§10, 2064-2071).
2. **Do not import any Hiwonder Python** — `espmax`, `SuctionNozzle`, `BusServo` are MicroPython modules that run on the ESP32 (§4, 364-438). Implement the §10 frames directly.
3. Frame builder: `bytes([0xAA,0x55,func,len]) + data + bytes([(~sum([func,len,*data])) & 0xFF])` (§10, 207, 1374-1383).
4. `move_to(x, y, z, ms)`: `func=0x03`, `data=struct.pack('<hhhH', int(x), int(y), int(z), int(ms))` (§10, 2152-2168). Coordinates in mm (§5, 123), +z up, front of arm at -y (§8, 936-951). Keep `sqrt(x²+y²) ≥ 50` and z ≤ 225 (§6, 458, 473).
5. `suction_on()`: `func=0x07, data=b'\x01'`. `suction_off()`: send `\x02`, sleep 0.2 s, send `\x03` (§10, 2382-2386, 2213).
6. **Completion:** no ack exists. Sleep `ms/1000 + settle`, then send `AA 55 13 00 EC`, read 11 bytes, unpack `<hhh` from bytes 4-9 and compare to the target with a tolerance of a few mm; treat mismatch as "refused/unreachable" (§10, 2865-2871, 205). Validate the reply checksum against both conventions (see (d)).
7. **Startup:** after opening the port, poll `read_xyz` for up to ~25 s before the first move (slave homes on `begin()`: §10, 886-887; plus the zip's 10 s sleep per web hunt).
8. Home first (`move_to(0, -163, 212, 2000)` or the value read back at boot), then **descend in z before moving in x/y** (§8, 282).
9. Ready-made reference drivers that already do all of the above and were tested on real hardware: my `connect4-robot/maxarm.py` (`https://raw.githubusercontent.com/aarnavshah12/connect4-robot/HEAD/maxarm.py`) and `https://github.com/Nu424/maxarm-python` (`pip install git+…`, `maxarmpy.MaxArm_ctl`). Either is a better starting point than the official `MaxArm_ctl.py` (set-vs-list bug, blocking 12-byte read, no timeout, no reply-checksum handling).
10. Fix in this project: any config with `BAUDRATE = 115200` for the PC link must become **9600** (§10, 120, 2361).
