# Cuda accelerated VP8 encoder based on libvpx
### Developed by Italtel and Universita' degli Studi di Milano, 2015-6
Pietro Paglierani, Giuliano Grossi, Federico Pedersini and Alessandro Petrini


## Project desctiption
As a contribution to the European Project "T-NOVA" (www.t-nova.eu), University of Milan - Department of Computer Science (www.di.unimi.it) and ITALTEL (www.italtel.com) have developed a GPU-accelerated VP8 encoder based on the existing libvpx 1.5.0 code. The main mission goal was to provide an accelerated encoder which outperformed - in terms of computational time - the open source vpxenc encoder, while retaining the same visual quality.

After evaluating different approaches to the problem, we found that the most promising strategy was to integrate the libvpx with a novel parallel motion estimation (ME) algorithm and provide a proper implementation specifically tailored for GP-GPU. We targeted modern NVidia graphic cards, and the parallel code was developed under the CUDA programming paradigm. This novel ME algorithm falls between the diamond search and the full search strategies since, during each iteration, several (up to 128) candidates inside the diamond area are evaluated at the same time, in order to exploit the massive parallelism exposed by GPU processors.
Three different ME CUDA kernels have been developed, each one having different features and performances, with the most accurate kernel featuring a complete splitmv (down to 4x4 blocks) search method and sub-pixel interpolation up to 1/4 pel motion vectors.

Regarding the performance, the slowest and most accurate kernel is able to boost the entire coding time up to 2x, while the speed-up obtained with the fastest MV algorithm goes up to 4.5x, at the expense of a small loss in visual quality.

Some preliminary tests were performed and the encodings were evaluated by following the submission recommendations stated on the Webm website. The results performed on different video resolutions (CIF, 720p and 1080p), on single pass and on single thread are available at:
```
<< add links >>
```
The tests were run on an Intel Xeon E5-2620 v3 @ 2.40 GHz, 64 GB ram, on Linux Ubuntu 14.05 64-bit, while the graphic card is a NVidia GTX980 with 4 GB of GDDR5 video memory.



## Build Instructions
Directory test_build has been already per-configured for building on x86_64 Linux, so that the
```
make
```
command would ideally be enough for compiling and linking.

Note that NVidia nvcc must be present in the system under the directory:
/usr/local/cuda-7.5/bin/
otherwise the files
```
test_build/libs-x86_64-linux-gcc.mk
test_build/examples-x86_64-linux-gcc.mk
```
must be modified accordingly accordingly.

It is possible to create a working makefile from scratch using the standard configure by following the instruction on BUILD_INSTR.txt


## Compiled binary
Compiled binary of vpxenc for 64-bit Linux and compute 3.5, 5.0 and 5.2 capable NVidia GPU cards is available for testing purpose under the bin directory.


## Usage
the command line options
```
--cuda-me=1
--cuda-me=2
--cuda-me=3
```
enables the GPU acceleration; mode 1 is the fastest and less accurate, while 2 is an intermediate level, and 3 is the most accurate.


## Contacts
Pietro Paglierani (ITALTEL) - email: pietro.paglierani@italtel.com

Giuliano Grossi (UNIMI) - email: grossi@di.unimi.it

Federico Pedersini (UNIMI) - email: pedersini@di.unimi.it

Alessandro Petrini (UNIMI) - email: alessandro.petrini@unimi.it


# DISCLAIMER

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
