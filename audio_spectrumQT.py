import audioop
import sys
import struct
import pyaudio
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtCore import QTimer

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FREQ_MAX = RATE // 2
FREQ_MIN = 20
CHUNK = 2 * RATE // FREQ_MIN
DELAY = 50


class WinForm(QWidget):
    def __init__(self):
        super(WinForm, self).__init__(parent=None)

        self.setWindowTitle("Audio Spectrum Analyzer")
        self.move(
            QApplication.desktop().screen().rect().center() - self.rect().center()
        )

        self.waveform = pg.PlotWidget(name="waveform")
        self.spectrum = pg.PlotWidget(name="spectrum")

        layout = QGridLayout()
        layout.addWidget(self.waveform)
        layout.addWidget(self.spectrum)

        self.setLayout(layout)

        self.waveform.setYRange(-5000, 5000)
        self.waveform.setXRange(0, CHUNK)
        self.waveform.showGrid(x=True, y=True, alpha=1)

        self.spectrum.setLogMode(x=True, y=False)
        self.spectrum.setYRange(-100, 1000)
        self.spectrum.setXRange(np.log10(FREQ_MIN), np.log10(FREQ_MAX))
        self.spectrum.showGrid(x=True, y=True, alpha=1)

        self.wv_x_axis = self.waveform.getAxis("bottom")
        self.wv_x_axis.setStyle(tickAlpha=0.5)

        self.wv_y_axis = self.spectrum.getAxis("left")
        self.wv_y_axis.setStyle(tickAlpha=0.5)

        self.sp_x_axis = self.spectrum.getAxis("bottom")
        self.sp_x_axis.setStyle(tickAlpha=0.5)

        self.sp_y_axis = self.spectrum.getAxis("left")
        self.sp_y_axis.setStyle(tickAlpha=0.5)

        sp_x_labels = [
            (np.log10(15), "15"),
            (np.log10(31), "31"),
            (np.log10(62), "62"),
            (np.log10(125), "125"),
            (np.log10(250), "250"),
            (np.log10(500), "500"),
            (np.log10(1000), "1k"),
            (np.log10(2000), "2k"),
            (np.log10(4000), "4k"),
            (np.log10(8000), "8k"),
            (np.log10(16000), "16k"),
        ]

        self.sp_x_axis.setTicks([sp_x_labels])
        self.sp_x_axis.setLabel("Frequency ", units="HZ")

        self.waveform.setTitle("waveform")
        self.spectrum.setTitle("spectrum")

        self.wv_tyle = {"color": "#FF0", "font-size": "20pt"}
        self.sp_tyle = {"color": "#0F0", "font-size": "20pt"}


class AudioStream:
    def __init__(self, m=0):
        self.p = pyaudio.PyAudio()

        try:
            self.stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            print("open microphone failed!\nError:", e)
            return

        self.x = np.arange(0, CHUNK)
        self.f = np.linspace(0, 22050, CHUNK)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(DELAY)

    def update(self):
        wf_data = self.stream.read(CHUNK)
        cnt = len(wf_data) // 2
        fmt = "%dh" % cnt
        shorts = struct.unpack(fmt, wf_data)
        self.m.waveform.plot(self.x, shorts, pen="c", clear=True)

        v_rms = audioop.rms(wf_data, 2)
        line_vms_y = np.linspace(v_rms, v_rms, cnt)
        self.m.waveform.plot(self.x, line_vms_y, pen="y", clear=False)

        self.f = np.fft.fftfreq(cnt, 1.0 / RATE)
        self.f = np.fft.fftshift(self.f)
        sp_data = np.abs((1.0 / cnt) * np.fft.fft(shorts))
        sp_data = np.fft.fftshift(sp_data)

        v_max = np.amax(sp_data)
        line_max_x = np.fft.fftfreq(cnt, 1.0 / RATE)
        line_max_y = np.linspace(v_max, v_max, cnt)

        self.m.spectrum.plot(self.f, sp_data, pen="m", clear=True)
        self.m.spectrum.plot(line_max_x, line_max_y, pen="g", clear=False)

        rms = 20 * np.log10(v_rms) if v_rms else 0
        self.m.wv_x_axis.setLabel("RMS:{:.0f}dB".format(rms), **self.m.wv_tyle)
        self.m.sp_x_axis.setLabel("MAX:{:.0f}".format(v_max), **self.m.sp_tyle)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.timer.stop()
        self.stream.close()


def main():
    app = QtWidgets.QApplication(sys.argv)
    a = AudioStream()
    if hasattr(a, "stream"):
        a.m = WinForm()
        a.m.show()
        sys.exit(app.exec_())
    else:
        app.exit()
        sys.exit()


if __name__ == "__main__":
    main()
