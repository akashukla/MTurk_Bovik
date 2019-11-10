$(function() {

    //Console logging (html)
    if (!window.console)
        console = {};
    
    var console_out = document.getElementById('console_out');
    var output_format = "jpg";

    //Slider init
    $("#slider-range-min").slider({
        range: "min",
        value: 100,
        min: 1,
        max: 100,
        slide: function(event, ui) {
            $("#jpeg_encode_quality").val(ui.value);
        },
        change: function(event, ui){
            $("jpeg_encode_button").click();
        }

    });
    $("#jpeg_encode_quality").val($("#slider-range-min").slider("value"));

    $( "#slider-range-min" ).on("slide", function( event, ui ) {
        $("#test").text("hello");

        var encodeButton = document.getElementById('jpeg_encode_button');
        var encodeQuality = document.getElementById('jpeg_encode_quality');
        var source_image = document.getElementById('source_image');
        var result_image = document.getElementById('result_image');
        if (source_image.src == "") {
            alert("You must load an image first!");
            return false;
        }

        var quality = parseInt(encodeQuality.value);        
        result_image.src = jic.compress(source_image,quality,output_format).src;
        
        result_image.onload = function(){
            var image_width=$(result_image).width(),
            image_height=$(result_image).height();
                
           result_image.style.display = "block";


        }
} );


});
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();           
            reader.onload = function (e) {
                $('#source_image').attr('src', e.target.result)
                $('#result_image').attr('src', e.target.result)

            };

            reader.readAsDataURL(input.files[0]);
        }
    }


