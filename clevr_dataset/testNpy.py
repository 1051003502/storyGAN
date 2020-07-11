class NpyDetector:
    def __init__(self):
        import numpy as np
        self.np=np
    def read(self,fileName):
        data=self.np.load(fileName,allow_pickle=True,encoding='latin1')
        print('')

if __name__=='__main__':
    detector = NpyDetector()
    detector.read('CLEVR_dict.npy')