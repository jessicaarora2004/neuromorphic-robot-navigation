import torch
import torch.nn as nn

import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate


class SpikingNavigation(nn.Module):
    

    def __init__(self,
                 num_steps=20,
                 beta=0.95):

        super().__init__()

        self.num_steps = num_steps
        self.spike_records = {}

        spike_grad = surrogate.fast_sigmoid()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.lif3 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.lif4 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad)

        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        spk_rec = []

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        num_steps = x.size(0)

        for step in range(num_steps):
            cur = x[step]

            cur = self.conv1(cur)
            spk1, mem1 = self.lif1(cur, mem1)
            self.spike_records["lif1"] = spk1.detach()

            cur = self.pool1(spk1)

            cur = self.conv2(cur)
            spk2, mem2 = self.lif2(cur, mem2)
            self.spike_records["lif2"] = spk2.detach()

            cur = self.pool2(spk2)

            cur = self.conv3(cur)
            spk3, mem3 = self.lif3(cur, mem3)
            self.spike_records["lif3"] = spk3.detach()

            cur = self.pool3(spk3)

            cur = cur.flatten(1)

            cur = self.fc1(cur)
            spk4, mem4 = self.lif4(cur, mem4)
            self.spike_records["lif4"] = spk4.detach()
            out = self.fc2(spk4)

            spk_rec.append(out)

        return torch.stack(spk_rec)