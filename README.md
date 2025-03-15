## ResNet Architecture Explanation
Submission by team महाराष्ट्र शासन (Maharashtra Shasan) in the Deep learning project competition. Team mates- Pranav Tushar Pradhan, Sahil Sarnaik, Rohan Gore

### Overall Structure

The ResNet model follows the general ResNet architecture, adapted for the CIFAR-10 dataset. It consists of the following main parts:

1.  **Initial Convolutional Layer:**  Preprocesses the input image.
2.  **Three Stages of Residual Blocks:** The core of the network, where the majority of the computation and feature learning happens. Each stage consists of multiple `BasicBlock` instances.
3.  **Global Average Pooling:**  Reduces the spatial dimensions of the feature maps.
4.  **Dropout Layer:** Helps reduce overfitting.
5.  **Fully Connected (Linear) Layer:**  Outputs the final classification scores.

### Detailed Layer Breakdown

The architecture can be visualized as follows:


| Layer Type             | Output Shape        | Details                                                                                                                                 |
|--------------------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Input                    | (3, 32, 32)         | CIFAR-10 Image (RGB, 32x32 pixels)                                                                                                     |
| **Initial Block**         |                     |                                                                                                                                          |
| Conv2d                  | (32, 32, 32)        | 3x3 kernel, stride=1, padding=1, `bias=False`                                                                                              |
| BatchNorm2d             | (32, 32, 32)        |                                                                                                                                          |
| ReLU                    | (32, 32, 32)        |                                                                                                                                          |
| **Stage 1 (7 Blocks)**    |                     |                                                                                                                                          |
| BasicBlock x 7          | (32, 32, 32)        | *Block 1:* `in_channels=32, out_channels=32, stride=1`<br> *Shortcut:*  Identity <br>  *Blocks 2-7:* Same as Block 1                       |
| **Stage 2 (7 Blocks)**    |                     |                                                                                                                                          |
| BasicBlock              | (64, 16, 16)        | *Block 1:* `in_channels=32, out_channels=64, stride=2`<br> *Shortcut:* 1x1 Conv2d (32 -> 64, stride=2), BatchNorm2d                       |
| BasicBlock x 6          | (64, 16, 16)        | *Blocks 2-7:* `in_channels=64, out_channels=64, stride=1`<br> *Shortcut:* Identity                                                         |
| **Stage 3 (7 Blocks)**    |                     |                                                                                                                                          |
| BasicBlock              | (190, 8, 8)         | *Block 1:* `in_channels=64, out_channels=190, stride=2`<br> *Shortcut:* 1x1 Conv2d (64 -> 190, stride=2), BatchNorm2d                      |
| BasicBlock x 6          | (190, 8, 8)         | *Blocks 2-7:*`in_channels=190, out_channels=190, stride=1`<br> *Shortcut:* Identity                                                        |
| **Final Layers**        |                     |                                                                                                                                          |
| AdaptiveAvgPool2d       | (190, 1, 1)         | Global Average Pooling, output size (1, 1)                                                                                                 |
| Flatten                 | (190)               | Reshape to (batch_size, 190)                                                                                                               |
| Dropout                 | (190)               |  Dropout with probability defined at initialization.                                                                                      |
| Linear                  | (10)                | Fully connected layer, `in_features=190`, `out_features=10` (for 10 CIFAR-10 classes)                                                        |

**1. Initial Convolutional Layer:**

*   **`nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)`:**
    *   Takes a 3-channel (RGB) input image.
    *   Applies 32 filters, each with a 3x3 kernel.
    *   Uses a stride of 1 (moves the filter one pixel at a time).
    *   Uses padding of 1 to maintain the spatial dimensions (32x32).  "Same" padding.
    *   `bias=False`:  The convolutional layer does *not* include a bias term. This is common practice when using batch normalization immediately after the convolution.
*   **`nn.BatchNorm2d(32)`:**  Applies batch normalization to the output of the convolutional layer.
*   **`F.relu(...)`:** Applies the ReLU activation function.

**2. Residual Stages:**

The core of the ResNet-  is composed of three stages, each containing a stack of `BasicBlock` instances.  The `BasicBlock` is the fundamental building block, and its details are explained in the previous section.

*   **Stage 1:**
    *   Consists of 7 `BasicBlock` instances.
    *   `in_channels=32`, `out_channels=32`, `stride=1` for all blocks in this stage.  The number of channels remains constant, and there's no downsampling.
*   **Stage 2:**
    *   Consists of 7 `BasicBlock` instances.
    *   The *first* block in this stage performs downsampling: `in_channels=32`, `out_channels=64`, `stride=2`.  The stride of 2 reduces the spatial dimensions by half (from 32x32 to 16x16), and the number of channels is doubled to 64.
    *   The remaining 6 blocks have `in_channels=64`, `out_channels=64`, `stride=1`.
*   **Stage 3:**
    *   Consists of 7 `BasicBlock` instances.
    *   The *first* block downsamples again: `in_channels=64`, `out_channels=190`, `stride=2`.  The spatial dimensions become 8x8, and the number of channels increases to 190.
    *   The remaining 6 blocks have `in_channels=190`, `out_channels=190`, `stride=1`.

**3. Global Average Pooling:**

*   **`F.adaptive_avg_pool2d(out, (1, 1))`:**
    *   Applies global average pooling. This takes the average of each feature map across its entire spatial extent.  If the input to this layer is of size (batch\_size, 190, 8, 8), the output will be (batch\_size, 190, 1, 1). This significantly reduces the number of parameters compared to using a fully connected layer directly after the convolutional layers.

**4. Flatten:**

* **`out.view(out.size(0), -1)`:** Reshapes the output of the global average pooling into vector by flattening the tensor. If input is of the shape (batch_size, 190, 1, 1), output would have shape (batch_size, 190)

**5. Dropout:**

*  **`self.dropout = nn.Dropout(dropout_rate)`:**
   *    Applies dropout which helps to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.

**6. Fully Connected Layer:**

*   **`nn.Linear(190, 10)`:**
    *   A fully connected layer that takes the 190-dimensional feature vector (from the global average pooling) as input and outputs a 10-dimensional vector.  Each of the 10 outputs corresponds to the score for one of the CIFAR-10 classes.

**Key Architectural Choices:**

*   **`BasicBlock`:**  The use of the `BasicBlock` with its shortcut connection is the defining feature of ResNets.
*   **Downsampling:** Downsampling (reducing spatial dimensions) is performed only at the *beginning* of stages 2 and 3, using a stride of 2 in the first convolutional layer of the first `BasicBlock` in each stage.
*   **Channel Increase:** The number of channels increases as the spatial dimensions decrease. This is a common pattern in CNNs.
*   **Global Average Pooling:**  Using global average pooling instead of flattening directly into a large fully connected layer significantly reduces the number of parameters, which helps to prevent overfitting and reduces computational cost.
*   **1x1 Convolutions in Shortcut:** When the dimensions change, 1x1 convolutions are used in the shortcut connection to match the dimensions.
* **Number of Filters**: The network starts with 32 channels and it remains as 32 after Stage 1, then becomes 64 at stage 2, finally increases to 190 after Stage 3.

This architecture achieves a good balance between depth (which allows the network to learn complex features) and the ability to train effectively (due to the residual connections). The specific choice of   layers (7 blocks per stage) is a design choice that likely balances performance and computational cost for the CIFAR-10 dataset.
