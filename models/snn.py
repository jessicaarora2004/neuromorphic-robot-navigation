import torch
import torch.nn as nn

import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate


class SpikingNavigation(nn.Module):

    def __init__(
        self,
        num_steps=20,
        beta=0.95
    ):
        super().__init__()

        self.num_steps = num_steps

        spike_grad = surrogate.fast_sigmoid()

        # -------------------------
        # Convolutional layers
        # -------------------------

        self.conv1 = nn.Conv2d(
            3, 16,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(2)

        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        self.conv2 = nn.Conv2d(
            16, 32,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(2)

        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        self.conv3 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.pool3 = nn.MaxPool2d(2)

        self.lif3 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        # -------------------------
        # Fully connected layers
        # -------------------------

        self.fc1 = nn.Linear(
            64 * 8 * 8,
            128
        )

        self.lif4 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(
            128,
            4
        )

        # Spike records
        self.spike_records = {}

    def forward(self, x):

        # x:
        # [batch, 3, 64, 64]

        # Convert images into temporal spike trains
        spike_data = spikegen.rate(
            x,
            num_steps=self.num_steps
        )

        # Reset spike records
        self.spike_records = {
            "lif1": [],
            "lif2": [],
            "lif3": [],
            "lif4": []
        }

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        output_record = []

        # -------------------------
        # Temporal processing
        # -------------------------

        for step in range(self.num_steps):

            cur = spike_data[step]

            # Layer 1
            cur = self.conv1(cur)

            spk1, mem1 = self.lif1(
                cur,
                mem1
            )

            self.spike_records["lif1"].append(
                spk1.detach()
            )

            cur = self.pool1(spk1)

            # Layer 2
            cur = self.conv2(cur)

            spk2, mem2 = self.lif2(
                cur,
                mem2
            )

            self.spike_records["lif2"].append(
                spk2.detach()
            )

            cur = self.pool2(spk2)

            # Layer 3
            cur = self.conv3(cur)

            spk3, mem3 = self.lif3(
                cur,
                mem3
            )

            self.spike_records["lif3"].append(
                spk3.detach()
            )

            cur = self.pool3(spk3)

            # -------------------------
            # Flatten
            # -------------------------

            cur = cur.flatten(1)

            # cur should be:
            # [batch, 4096]

            # Fully connected layer
            cur = self.fc1(cur)

            spk4, mem4 = self.lif4(
                cur,
                mem4
            )

            self.spike_records["lif4"].append(
                spk4.detach()
            )

            # Output layer
            out = self.fc2(spk4)

            output_record.append(out)

        # Convert lists to tensors

        for key in self.spike_records:

            self.spike_records[key] = torch.stack(
                self.spike_records[key]
            )

        return torch.stack(output_record)