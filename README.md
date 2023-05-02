Download Link: https://assignmentchef.com/product/solved-cmu11485-homework-2-part-1-mytorch
<br>
<ul>

 <li><strong>Overview:</strong>

  <ul>

   <li><strong>Convolutional Network Components</strong>: Using the Autograd framework to implement the forward and backward passes of a 1D convolutional layer and a flattening layer.</li>

   <li><strong>CNNs as Scanning MLPs</strong>: Two questions on converting a linear scanning MLP to a CNN <strong>– Implementing a CNN Model</strong>: You know the drill.</li>

  </ul></li>

 <li><strong>Directions:</strong>

  <ul>

   <li>We estimate this assignment may take up to 5 hours; it is <em>significantly </em>shorter and simpler than not just HW1P1, but also the previous version of the assignment. We told you future homeworks would now be easier thanks to autograd, and we meant it.</li>

  </ul></li>

</ul>

∗ If you’re truly stuck, post on Piazza or come to OH; don’t take more time than you need to.

<ul>

 <li>You are required to do this assignment using Python3. Do not use any auto-differentiation toolboxes (PyTorch, TensorFlow, Keras, etc) – you are only permitted and recommended to vectorize your computation using the NumPy library.</li>

 <li>If you haven’t already been doing so, use pdb to debug your code and please PLEASE Google your error messages before posting on Piazza. <strong>– </strong>Note that Autolab uses <a href="https://numpy.org/doc/1.18/">numpy v1.18.1</a></li>

</ul>

<h1>Introduction</h1>

In this assignment, you will continue to develop your own version of PyTorch, which is of course called MyTorch (still a brilliant name; a master stroke. Well done!). In addition, you’ll convert two scanning MLPs to CNNs and build a CNN model.

<h2>Homework Structure</h2>

Below is a list of files that are <strong>directly relevant </strong>to hw2.

<strong>IMPORTANT: </strong>First, copy the highlighted files/folders from the HW2P1 handout over to the corresponding folders that you used in hw1.

<strong>NOTE: We recommend you make a backup of your hw1 files before copying everything over, just in case you break code or want to revert back to an earlier version.</strong>

<table width="97">

 <tbody>

  <tr>

   <td width="41">hw2_au</td>

   <td width="14">to</td>

   <td width="42"> </td>

   <td width="0"></td>

  </tr>

  <tr>

   <td colspan="3" rowspan="7" width="97">runner.py test_cnn.py test_conv.pydata weights</td>

   <td width="0"></td>

  </tr>

 </tbody>

</table>

handout autograder grader……………………………………………..Copy over this entire folder

test_scanning.py

ref_result

mytorch…………………………………………………………………MyTorch library nn……………………………………………………………..Neural Net-related files activations.py

conv.py………………………………………[Question 1] Convolutional layer objects functional.py linear.py sequential.py

tensor.py sandbox.py create_tarball.sh grade.sh

hw2……………………………………………………………Copy over this entire folder hw2.py………………………………………………[Question 3] Building a CNN model mlp_scan.py…………………………….[Question 2] Converting Scanning MLPs to CNNs stubs.py………………….Contains new code for functional.py that you’ll need to copy over <strong>Next, </strong>copy and paste the following code stubs from hw2/stubs.py into the correct files.

<ol>

 <li>Copy Conv1d(Function) into nn/functional.py.</li>

 <li>Copy get_conv1d_output_size() into nn/functional.py.</li>

 <li>Copy Tanh(Function) and Sigmoid(Function) into nn/functional.py.</li>

 <li>Copy Tanh(Module) and Sigmoid(Module) into nn/activations.py.</li>

</ol>

Note that we’re giving you the fully completed Tanh and Sigmoid code as a peace offering after the Great Battle of HW1P1.

<h2>0.1         Running/Submitting Code</h2>

This section covers how to test code locally and how to create the final submission.

<strong>0.1.1            Running Local Autograder</strong>

Run the command below to calculate scores and test your code locally.

./grade.sh 2

If this doesn’t work, converting <a href="https://en.wikipedia.org/wiki/Newline">line-endings</a>              may help:

sudo apt install dos2unix dos2unix grade.sh

./grade.sh 2

If all else fails, you can run the autograder manually with this:

python3 ./autograder/hw2_autograder/runner.py

<strong>0.1.2           Running the Sandbox</strong>

We’ve provided sandbox.py: a script to test and easily debug basic operations and autograd.

<strong>Note: We will not provide new sandbox methods for this homework. You are required to write your own from now onwards.</strong>

python3 sandbox.py

<strong>0.1.3            Submitting to Autolab</strong>

<strong>Note: You can submit to Autolab even if you’re not finished yet. You should do this early and often, as it guarantees you a minimum grade and helps avoid last-minute problems with Autolab.</strong>

Run this script to gather the needed files into a handin.tar file:

./create_tarball.sh

If this crashes (with some message about a hw4 folder) use dos2unix on this file too.

You can now upload handin.tar to <a href="https://autolab.andrew.cmu.edu/courses/11485-f20/assessments">Autolab</a>

<h1>1          Convolutional Network Components</h1>

Let’s get down to business

To write a CNN

<h2>1.1         Conv1d</h2>

If you have not already, make sure to copy over files and code as instructed on the previous page.

In this problem, you will be implementing the forward and backward of a 1-dimensional convolutional layer.

You will only need to implement the 1D version for this course; the 2D version will be part of the optional hw2 bonus, which will be released at the same time as hw3.

Important note: You will be <strong>required </strong>to implement <strong>Conv1d </strong>using a subclass of Function. To repeat, you may not automate backprop for <strong>Conv1d</strong>.

We are requiring this over automating backprop for these reasons:

<ol>

 <li>Implementing backprop for CNNs teaches valuable lessons about convolutions and isolating influences of their weights.</li>

 <li>We hope it’ll cement your understanding of autograd, as you do what it would do, in tracing paths and accumulating gradients.</li>

 <li>The actual Torch implements a subclass of Function. It’s almost certainly faster and takes MUCH less memory than maintaining dozens of tensors and instructions for each individual operation. You’ll see what we mean; picture storing every 2 tensors for every operation of a 100 Conv layer network.</li>

 <li>Finally, the lec 10 slides contain detailed pseudocode for the forward/backward to make things easier.</li>

</ol>

<strong>Again, if you automate backprop, you will not receive credit for this problem, even if you pass the test on autolab.</strong>

<h3>1.1.1         Conv1d.forward()</h3>

First, in nn/functional.py, complete the get_conv1d_output_size() function.

You’ll need this function when writing the Conv1d forward pass in order to calculate the size of the output data (output_size). Note that this function should NOT add to the computational graph. Because we’re not implementing padding or dilation, we can implement this simplified formula<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>:

[(input_size − kernel_size<sub>)//</sub>stride<sub>] + 1</sub>

Next, in nn/functional.py, complete Conv1d.forward(). This is the 1D convolutional layer’s forward behavior in the computational graph.

The pseudocode for this is in the slides on CNNs.

For explanations of each variable in Conv1d, read the comments in both functional.Conv1d and nn.conv.Conv1d. Also, read the <a href="https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html">excellent Torch documentation</a> for it.

Finally, note that we’ve already provided the complete user-facing Conv1d(Module) object in nn/conv.py.

<h3>1.1.2         Conv1d.backward()</h3>

In nn/functional.py, complete Conv1d.backward().

This will be uncannily similar to the forward’s code (I wonder why… ).

Again, pseudocode in the slides.

<h2>1.2         Flatten.forward()</h2>

In nn/conv.py, complete Flatten.forward(). Note that this is in the conv.py file, not in functional.py.

This layer is often used between Conv and Linear layers, in order to squish the high-dim convolutional outputs into a lower-dim shape for the linear layer. For more info, see the <a href="https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html">torch documentation</a> and the example we provided in the code comments.

<strong>Hint: </strong>This can be done in one line of code, with no new operations or (horrible, evil) broadcasting needed.

<strong>Bigger Hint: </strong>Flattening is a subcase of reshaping. np.prod() may be useful.

<h1>2          Converting Scanning MLPs to CNNs</h1>

In these next two problems, you will be converting the weights of a 3-layer scanning MLP into the weights of a 3-layer CNN. The goal is to demonstrate the equivalence between scanning MLPs and CNNs.

Below is a description of the MLP you’ll be converting.

<h2>Background</h2>

<table width="625">

 <tbody>

  <tr>

   <td colspan="3" width="31"> </td>

   <td colspan="3" width="31"> </td>

   <td colspan="2" width="25"> </td>

   <td colspan="3" width="39"> </td>

   <td colspan="3" width="33"> </td>

   <td colspan="2" width="16"> </td>

   <td colspan="3" width="22"> </td>

   <td colspan="2" width="33"> </td>

   <td width="15"> </td>

   <td colspan="3" width="38"> </td>

   <td width="25"> </td>

   <td colspan="4" width="36"> </td>

   <td colspan="2" width="24"> </td>

   <td colspan="4" width="39"> </td>

   <td colspan="2" width="32"> </td>

   <td width="10"> </td>

   <td colspan="2" width="28"> </td>

   <td width="23"> </td>

   <td colspan="2" width="26"> </td>

   <td width="24"> </td>

   <td colspan="3" width="33"> </td>

   <td colspan="2" width="40"> </td>

  </tr>

  <tr>

   <td colspan="2" width="24">info</td>

   <td colspan="3" width="25">you</td>

   <td colspan="2" width="31">need</td>

   <td colspan="2" width="16">to</td>

   <td colspan="2" width="30">solve</td>

   <td rowspan="2" width="4"> </td>

   <td width="21">the</td>

   <td colspan="2" width="17">up</td>

   <td colspan="3" width="24">com</td>

   <td colspan="2" width="20">ing</td>

   <td colspan="2" width="33">prob</td>

   <td colspan="2" width="27">lems.</td>

   <td colspan="2" width="36">Also,</td>

   <td colspan="2" width="20">We</td>

   <td colspan="4" width="41">highly</td>

   <td colspan="2" width="19">rec</td>

   <td colspan="2" width="20">om</td>

   <td colspan="2" width="32">mend</td>

   <td colspan="2" width="24">you</td>

   <td colspan="2" width="37">draw</td>

   <td width="10">a</td>

   <td width="16">sim</td>

   <td width="24">ple</td>

   <td width="11">vi</td>

   <td width="10">su</td>

   <td width="12">al</td>

   <td width="16">iza</td>

   <td width="24">tion</td>

  </tr>

  <tr>

   <td width="13">of</td>

   <td colspan="3" width="29">this.</td>

   <td colspan="2" width="21">See</td>

   <td colspan="2" width="25">the</td>

   <td colspan="2" width="16">lec</td>

   <td width="23">ture</td>

   <td colspan="2" width="29">slides</td>

   <td colspan="3" width="22">for</td>

   <td colspan="2" width="15">ex</td>

   <td width="15">am</td>

   <td colspan="2" width="33">ples.</td>

   <td width="19">We</td>

   <td colspan="2" width="19">left</td>

   <td width="25">this</td>

   <td width="4"> </td>

   <td width="15">ex</td>

   <td width="9">er</td>

   <td colspan="2" width="22">cise</td>

   <td colspan="2" width="15">to</td>

   <td colspan="2" width="24">you</td>

   <td colspan="2" width="15">so</td>

   <td width="27">you’d</td>

   <td width="10"> </td>

   <td colspan="2" width="28">fully</td>

   <td width="23">un</td>

   <td width="10">der</td>

   <td colspan="2" width="39">stand</td>

   <td colspan="2" width="22">this</td>

   <td colspan="2" width="28">prob</td>

   <td width="24">lem.</td>

  </tr>

  <tr>

   <td width="13"></td>

   <td width="11"></td>

   <td width="6"></td>

   <td width="10"></td>

   <td width="8"></td>

   <td width="16"></td>

   <td width="18"></td>

   <td width="7"></td>

   <td width="10"></td>

   <td width="7"></td>

   <td width="28"></td>

   <td width="4"></td>

   <td width="23"></td>

   <td width="9"></td>

   <td width="9"></td>

   <td width="8"></td>

   <td width="7"></td>

   <td width="12"></td>

   <td width="4"></td>

   <td width="15"></td>

   <td width="16"></td>

   <td width="13"></td>

   <td width="20"></td>

   <td width="11"></td>

   <td width="11"></td>

   <td width="26"></td>

   <td width="4"></td>

   <td width="15"></td>

   <td width="9"></td>

   <td width="8"></td>

   <td width="15"></td>

   <td width="10"></td>

   <td width="7"></td>

   <td width="14"></td>

   <td width="11"></td>

   <td width="12"></td>

   <td width="5"></td>

   <td width="27"></td>

   <td width="11"></td>

   <td width="14"></td>

   <td width="13"></td>

   <td width="22"></td>

   <td width="10"></td>

   <td width="15"></td>

   <td width="20"></td>

   <td width="10"></td>

   <td width="10"></td>

   <td width="12"></td>

   <td width="16"></td>

   <td width="23"></td>

  </tr>

 </tbody>

</table>

Note that you won’t need to program or train this MLP anywhere; this is just context and background

The MLP is scanning a single observation of time series data, sized (1, 128, 24) (note: batch size of 1).

In other words, this observation has 128 time instants, which are each a 24-dimensional vector.

This is the architecture for the MLP:

[Flatten(), Linear(8 * 24, 8), ReLU(), Linear(8, 16), ReLU(), Linear(16, 4)]

[Flatten()] # after scanning is completed, all outputs are concatenated then flattened <strong>Some architecture notes:</strong>

<ul>

 <li>Assume all bias values are 0. Don’t worry about converting or setting bias vectors; focus on the weights.</li>

 <li>This architecture will be (nominally) identical in both Problem 2.1 and 2.2, although there is a key difference in their weight values, which we’ll explain in Problem 2.2.</li>

</ul>

For each forward pass, the network will receive 8 adjacent vectors at a time: a tensor sized (1, 8, 24). So the network’s first scan will be over training_data[:, 0:8, :]. The network will then flatten this input into a size (1, 192) tensor<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> and then pass it into the first linear layer. Because the final layer has 4 neurons, the network will produce a (1, 4) output for each forward pass.

After each forward pass, the MLP will “stride” forward 4 time instants (i.e. the second scan is over training_data[:, 4:12, :]), until it scans the last possible vectors given its stride and input layer size. Note that the network will not pad the data anywhere.

This means the network will stride 31 times and produce a concatenated output of size (1, 4, 31)<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>. After flattening, the output size will be (1, 124) (because <sub>4 </sub>× <sub>31 = 124</sub>).

<strong>Summary</strong>:

<ul>

 <li>3 layer MLP with # neurons [8, 16, 4].</li>

 <li>ReLUs between Linear layers</li>

 <li>Data we are scanning is size (1, 128, 24)</li>

 <li>Network scans 8 inputs at a time ((1, 8, 24) input tensor), producing an output of (1, 4).</li>

 <li>After each forward pass, strides 4 time instants.</li>

 <li>Outputs are concatenated into a tensor sized (1, 4, 31), and after flattening will be size (1, 124) <strong>Goal: convert this scanning MLP to a CNN.</strong></li>

</ul>

<h2>2.1         Converting a Simple Scanning MLP [10 Points]</h2>

In hw2/mlp_scan.py, complete CNN_SimpleScanningMLP.

There’s only two methods you need to complete: __init__() and init_weights(). You’ll need to finish both before you can run the autograder. Feel free to do them in either order; they’re both pretty short (but can be conceptually difficult, as is normal for IDL).

The comments in the code will tell you what to do and what each object does.

As you’ll quickly see, the challenge will be in determining what numbers to use when initializing the Conv1d layers.

Here are a lot of hints.

<strong>General Tips</strong>:

<ul>

 <li>Make sure to not add nodes to the comp graph. You should know what this means by now.</li>

 <li>Tensor settings matter; autograder will check for them.</li>

 <li>You’re allowed to read the autograder files (obviously we haven’t given away answers, but it may help you get unstuck).</li>

</ul>

<strong>Pseudocode for init_weights()</strong>:

<ol>

 <li>Reshape each Linear weight matrix into (out_channel, kernel_size, in_channel).</li>

 <li>Transpose the <u>axes </u>back into these shapes: (out_channel, in_channel, kernel_size). (Note: .T will not work for this step)</li>

 <li>Set each array as the params of each conv layer.</li>

</ol>

Again, don’t worry about biases for this problem AND the next one. If you ask about biases at OH or Piazza, we’ll know you didn’t read the writeup properly and you’ll instantly lose a lot of street cred<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a>.

<h2>2.2         Converting a Distributed Scanning MLP</h2>

This problem is very similar to the previous, except the MLP is now a <strong>shared-parameter network </strong>that captures a distributed representation of the input.

All background/context is still the same (data size, stride size, network components, etc). Even the overall architecture of the MLP is identical. This includes the number of neurons in each layer (still [8, 16, 4]).

But the only difference is that <strong>many of the neurons within each layer share parameters with each other</strong>.

We’ve illustrated the parameter-sharing pattern below.

Figure 1: The Distributed Scanning MLP network architecture

Each time instant <em><sub>t </sub></em>is a single line on the bottom. Each circle is a neuron, and each edge is a connection. So “Layer 1” would be the edges between the inputs and the neurons just above the inputs.

<strong>Key point</strong>: within each layer, the neurons with the same color share the same weights. Note that this coloring scheme only applies within each layer; neurons with the same/similar colors in other layers do <strong>not </strong>share weights.

If this illustration is unfamiliar or confusing, please rewatch the explanations of distributed scanning MLPs in the lectures. Verbal explanations make this much clearer.

Try to picture what the shape of each weight matrix would look like, especially in comparison to those in the previous problem.

In hw2/mlp_scan.py, complete the class CNN_DistributedScanningMLP.

Most of the code will be identical; you need to complete the same two methods as the previous problem. But there are a few differences, described below:

<ol>

 <li>In init_weights(), you’ll need to first somehow slice the weight matrix(s) to account for the shared weights. Afterwards, the rest of the code will be the same.</li>

 <li>In __init__(), the model components will be identical, but the params of each Conv1d (i.e. kernel_size, stride, etc) will not be.</li>

</ol>

<h1>3          Build a CNN model</h1>

Finally, in hw2/hw2.py, implement a CNN model.

Figure 2: CNN Architecture to implement.

Remember that we provided you the Tanh and Sigmoid code; if you haven’t already, see earlier instructions for copying and pasting them in.

Previously for this problem, students would have to implement step, forward, backward, etc. But thanks to your hard work in this homework and especially in HW1P1, all you have to do is initialize the layer objects like in the architecture pictured above, and you’re done.

The only tedious thing is calculating the input size of the final Linear layer.

Why is this tedious? Notice that the input to the final Linear layer is flattened into size

(batch_size, conv3_num_channels * conv3_output_size). The batch size and the number of channels for conv3 are obvious, but what’s that second term in the second dimension: the output size?

The network input x will be shaped (batch_size, num_input_channels, input_width), where input_width=60. But remember from your Conv implementation that each Conv forward will change the size of the data’s final dimension (input_width -&gt; output_width). So the size of this final dimension will change every time it passes through a Conv layer. Here, it will change 3 times. So you’ll need to calculate this in order to initialize the final linear layer properly.

On the bottom of the same file, complete get_final_conv_output_size(). Given the initial input_size, this method should iterate through the layers, check if the layer is a convolutional layer, and update the current size of the final dimension using the get_conv1d_output_size() function you implemented earlier. This method will take the current output_size and the params of the layer to calculate the next output_size. After iterating through all the layers, you’ll have the output_size for the final Conv1d layer.

(Above is essentially pseudocode for this method; don’t miss it.)

<strong>We ask you to implement this because you may want to modify it and use it for HW2P2</strong>.

Given any CNN-based architecture, this function should return the correct output size of the final layer before flattening. While you can assume only Conv1d layers will matter for now, if you do reuse this you may need to modify it to account for other layer types.

<a href="#_ftnref1" name="_ftn1">[1]</a> If you want to reuse this function in hw2p2, see the Conv1d torch docs for the full formula that includes padding, dilation, etc.

<a href="#_ftnref2" name="_ftn2">[2]</a> batch_size first

<a href="#_ftnref3" name="_ftn3">[3]</a> This is tricky; why 31? You may expect the output size to be 32, because <sub>128/4=32</sub>. Try this simple example: draw out an MLP that sees 3 observations at a time, and strides by 2, over an observation of 16 time instants. How many forward passes will it perform? Try to come up with a formula that would tell you the number of outputs. Hint: remember that CNNs and scanning MLPs are equivalent.

<a href="#_ftnref4" name="_ftn4">[4]</a> The cool kids will hear about it and won’t let you sit at their table anymore.