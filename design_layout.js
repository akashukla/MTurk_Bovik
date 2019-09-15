<!-- You must include this JavaScript file -->
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<link href="//netdna.bootstrapcdn.com/bootstrap/3.1.0/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
<script src="//netdna.bootstrapcdn.com/bootstrap/3.1.0/js/bootstrap.min.js"></script>
<script src="//code.jquery.com/jquery-1.11.1.min.js"></script>

    <div class="container">
      <hr/>
      <div class="row">
        <div class="col-xs-6">
          <div class="range">
            <input type="range" name="range" min="1" max="100" value="50" onchange="range.value=value">
            <output id="range">50</output>
          </div>
        </div>
    </div>

<!-- For the full list of available Crowd HTML Elements and their input/output documentation,
      please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

<!-- You must include crowd-form so that your task submits answers to MTurk -->
<crowd-form answer-format="flatten-objects">

    <!-- The crowd-classifier element will create a tool for the Worker to select the
           correct answer to your question.

          Your image file URLs will be substituted for the "image_url" variable below 
          when you publish a batch with a CSV input file containing multiple image file URLs.
          To preview the element with an example image, try setting the src attribute to
          "https://s3.amazonaws.com/cv-demo-images/two-birds.jpg" -->
    <crowd-image-classifier 
        src="${image_url}"
        categories="['Yes', 'No', 'Cannot Determine']"
        header="Select the highest compression level at which the image is still clear"
        name="acceptable-image-compression">

       <!-- Use the short-instructions section for quick instructions that the Worker
              will see while working on the task. Including some basic examples of 
              good and bad answers here can help get good results. You can include 
              any HTML here. -->
        <short-instructions>

        <!-- Use the full-instructions section for more detailed instructions that the 
              Worker can open while working on the task. Including more detailed 
              instructions and additional examples of good and bad answers here can
              help get good results. You can include any HTML here. -->
        <full-instructions header="Does this image contain the object of interest?">
            <h2>Detailed instructions</h2>
            <p>Instructions for determining if the image contains the object:</p>
            <ol>
	        <li>Begin by carefully examining the entire image</li>
        	<li>Next determine if the object of interest occurs in the image</li>
	        <li>If it the object exists in the image, select that option; otherwise, indicate it does not</li>
            </ol>
        </full-instructions>
    </crowd-image-classifier>
</crowd-form>
