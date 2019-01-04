import os
import pycuda.driver as cuda
import pycuda.autoinit

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cuda.init()


class about_cuda_devices():
    def __init__(self):
        pass

    def num_devices(self):
        return cuda.Device.count()

    def devices(self):
        for i in range(cuda.Device.count()):
            print(cuda.Device(i).name(), '(Id: {})'.format(i))

    def mem_info(self):
        available, total = cuda.mem_get_info()
        print('Available: {:2.2%} GB\nTotal: {:2.2f}'.
              format(available / 1e9, total / 1e9))

    def attributes(self, device_id=0):
        return cuda.Device(device_id).get_attributes()

    def __repr__(self):
        string = ""
        string += ('{} device(s) found:\n'.format(self.num_devices()))
        for i in range(self.num_devices()):
            string += '\t {}), {}\n'.format(i, cuda.Device(i).name())
            string += '\t\t Memory: {} GB\n'.format(cuda.Device(i).total_memory() / 1e9)
        return string


print(about_cuda_devices().__repr__())
