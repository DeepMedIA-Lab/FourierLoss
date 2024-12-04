import numpy as np
import cv2 as cv

import torch 
import torch.nn as nn
        
        
class FourierDescriptors:
    def __init__(self, N, modes, descriptor):
        self.N = N
        self.modes = modes
        self.descriptor = descriptor
        
    def calculate_descriptors(self, y_true):
        num_of_classes = y_true.max()
        descriptors = []
        cnt = []
        if num_of_classes != 0:
            for c in range(1, num_of_classes+1):
                y_c = np.uint8(y_true==c)
                descriptors_c = []

                contours, _ = cv.findContours(y_c.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                contours_new = np.array(max(contours, key=cv.contourArea))

                cnt.append(contours)    

                descriptors_c.append(self.calculate_fourier_descriptors(contours_new, c-1))
                descriptors.append(np.array(descriptors_c))
        else:
            cnt.append(0)
            descriptors_c = []
            
            descriptors_c.append([0 for infex in range(self.N[0])])
            descriptors.append(np.array(descriptors_c))
        return descriptors, cnt

    def calculate_fourier_descriptors(self, contour, c):
        if self.modes[c]=='center':
            return self.calculate_fourier_descriptors_center(contour, c)
        elif self.modes[c]=='angle':
            return self.calculate_fourier_descriptors_angle(contour, c)

    def calculate_fourier_descriptors_center(self, contour, c):
        center = self.calculate_center(contour)
        delta = []
        l = []
        for i in range(1, len(contour)+1):
            point1 = (contour[i-1][0][1], contour[i-1][0][0])
            point2 = (contour[i%len(contour)][0][1], contour[i%len(contour)][0][0])
            d1 = np.sqrt((point1[0]-center[0])**2+(point1[1]-center[1])**2)
            d2 = np.sqrt((point2[0]-center[0])**2+(point2[1]-center[1])**2)
            delta.append(d1-d2)
            d3 = np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
            l.append(d3)
        for i in range(1, len(l)):
            l[i] += l[i-1]
        A = []
        for i in range(1, self.N[c]+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_center(self, contour):
        x, y, num = 0, 0, 0
        for pixel in contour:
            num += 1
            x += pixel[0][1]
            y += pixel[0][0]
        return x/num, y/num
    
    def calculate_fourier_coefficients(self, k, l, delta):
        a, b = 0, 0
        L = l[-1]
        for i in range(len(l)):
            if delta[i]!=0:
                a += delta[i]*np.sin((2*np.pi*k*l[i])/L)
                b += delta[i]*np.cos((2*np.pi*k*l[i])/L)
        a = a/(k*np.pi)
        b = -b/(k*np.pi)
        if self.descriptor=='harmonic_amplitude':
            return np.sqrt(a*a+b*b)
        elif self.descriptor=='phase_angle':
            return np.arctan(b/(a+1e-10)) + np.pi
        

    

class fourier_loss(nn.Module):    

    def __init__(self, device,N=[2],mode=['center']):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([3., 1., 0.5, 0.25, 0.25]))
        self.weights.requires_grad = True
        self.device = device
        self.weight_coeff = [0.05, 0.1, 0.5]
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.N = N
        self.fd_obj = FourierDescriptors(N, mode, 'harmonic_amplitude')
        
    def forward(self, outputs, targets, golds_desc):

        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().detach().numpy()
        preds_desc = [self.fd_obj.calculate_descriptors(pred.astype(int))[0][0][0] for pred in preds]

        weights = self.weights[:self.N[0]]
        beta = weights * torch.abs(golds_desc - torch.tensor(np.array(preds_desc)))
        ce_coeff = 1 + beta * self.weight_coeff[0]
        ce_coeff = torch.sum(ce_coeff, dim=1).to(self.device)
        ce_loss = self.ce(outputs, targets)
        ce_loss = torch.mean(ce_loss,dim = [1,2])
        
        fourier_loss = (ce_coeff * ce_loss).mean()
        return fourier_loss
