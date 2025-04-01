import neopixel
import time
from machine import Pin, I2C
from as7341 import AS7341, AS7341_MODE_SPM
# For the NeoPixel LED
pixel_pin = Pin(28, mode=Pin.OUT)  # NeoPixel LED connected to Pin 28
num_pixels = 1  # Number of NeoPixels
pixels = neopixel.NeoPixel(pixel_pin, num_pixels)  # Create NeoPixel object
def set_color(color):
    """Set the color of the NeoPixel LED."""
    pixels[0] = color
    pixels.write()
def turn_off():
    """Turn off the NeoPixel LED."""
    pixels.fill((0, 0, 0))
    pixels.write()
# For the AS7341 light sensor
class Sensor:
    def __init__(self, atime=100, astep=999, gain=8, i2c=I2C(1, scl=Pin(27), sda=Pin(26))):
        """Initialize the sensor with specified settings."""
        self.i2c = i2c
        addrlist = " ".join(["0x{:02X}".format(x) for x in i2c.scan()])
        print("Detected devices at I2C-addresses:", addrlist)
        self.sensor = AS7341(i2c)
        if not self.sensor.isconnected():
            raise Exception("Failed to contact AS7341, terminating")
        self.sensor.set_measure_mode(AS7341_MODE_SPM)
        self.sensor.set_atime(atime)
        self.sensor.set_astep(astep)
        self.sensor.set_again(gain)
    def all_channels(self):
        """Read and return all spectral channels including the CLR channel."""
        channel_names = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'CLR']
        self.sensor.start_measure("F1F4CN")
        f1, f2, f3, f4, clr1, nir1 = self.sensor.get_spectral_data()
        self.sensor.start_measure("F5F8CN")
        f5, f6, f7, f8, clr2, nir2 = self.sensor.get_spectral_data()
        channels = [f1, f2, f3, f4, f5, f6, f7, f8, clr1]  # Assuming clr1 == clr2 for simplicity
        for name, value in zip(channel_names, channels):
            print(f"{name}: {value}")
        # Optionally, return the channels and CLR if needed elsewhere
        return channels
# Main execution flow
def main():
    # Blink the NeoPixel with white color
    set_color((255, 255, 255))
    time.sleep(2)  # Keep the LED on for 2 seconds
    turn_off()
    # Initialize and read from the AS7341 sensor
    try:
        sensor = Sensor()
        sensor.all_channels()  # This will print channel names with their values
    except Exception as e:
        print(str(e))
if __name__ == "__main__":
    main()