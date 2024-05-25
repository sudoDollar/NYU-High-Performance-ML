## C++ Chatbot using Pytorch C++ (LibTorch) and Torchscript model

**File Name:** nonPython_chatbot.cc

## Code Details:
**Classes:**\
class Voc => Vocabulary Class to store and build word/index map. \
class Utils => Helper class to process strings and dataset. \
class TorchModel => Interface for accessing loaded model.

main() method => Chatbot using above classes


## How to Run:
**Step 1:** **Install Requirements:**
1. Download and Extract [LibTorch (Ubuntu)](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcpu.zip) OR [LibTorch (MacOS ARM)](https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.2.1.zip)
2. Cmake: *brew install cmake* (MacOS) or *sudo apt install cmake* (Ubuntu)
3. Code Files: nonPython_chatbot.cc, CMakeLists.txt, vocab.csv, scripted_chatbot_cpu.pt
4. C++ version >= 17
5. For MacOS, you may need to install libomp via *brew install libomp*

**Step 2:** **Building:**\
Within the directory where nonPython_chatbot.cc and CMakeLists.txt are present, run following commands:
```
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . --config Release
```
For example:
```
cmake -DCMAKE_PREFIX_PATH="/root/snap/HPML/libtorch"
cmake --build . --config Release
```

**Step 3:** **Run**:\
Run executable file with following command:
```
./nonPython_chatbot <path/to/scripted_chatbot_cpu.pt> <path/to/vocab.csv>
```
For example (if all files are in same directory):
```
./nonPython_chatbot scripted_chatbot_cpu.pt vocab.csv
```
OR
```
 ./nonPython_chatbot "/root/snap/HPML/Lab3/scripted_chatbot_cpu.pt" "/root/snap/HPML/Lab3/vocab.csv"
```

**Note:** Code has been tested on MacOS with M1 and Ubuntu with Intel CPUs and working fine. Not tested on other platforms or other CPUs.

**P.S:** While interacting with Bot, enter "quit" or "q" to exit the program.
