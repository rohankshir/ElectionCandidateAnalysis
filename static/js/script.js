google.charts.load('current', {packages: ['corechart', 'bar']});
var opts = {
  lines: 13 // The number of lines to draw
, length: 28 // The length of each line
, width: 14 // The line thickness
, radius: 42 // The radius of the inner circle
, scale: 1 // Scales overall size of the spinner
, corners: 1 // Corner roundness (0..1)
, color: '#000' // #rgb or #rrggbb or array of colors
, opacity: 0.25 // Opacity of the lines
, rotate: 0 // The rotation offset
, direction: 1 // 1: clockwise, -1: counterclockwise
, speed: 1 // Rounds per second
, trail: 60 // Afterglow percentage
, fps: 20 // Frames per second when using setTimeout() as a fallback for CSS
, zIndex: 2e9 // The z-index (defaults to 2000000000)
, className: 'spinner' // The CSS class to assign to the spinner
, top: '50%' // Top position relative to parent
, left: '50%' // Left position relative to parent
, shadow: false // Whether to render a shadow
, hwaccel: false // Whether to use hardware acceleration
, position: 'absolute' // Element positioning
}
var target = document.getElementById('chart_div')
//google.charts.setOnLoadCallback(drawBasic);


function drawBasic(probability, query) {

    console.log(probability)
      var data = google.visualization.arrayToDataTable([
          ['Nominee', 'Probability', { role: 'style' }],
          ['Hillary Clinton', probability[0], 'color: #1E90FF'],
          ['Donald Trump', probability[1], 'color: #DC143C'],
      ]);

      var options = {
        title: 'Probability of Nominee saying: "' + query + '" ',
        chartArea: {width: '50%'},
        hAxis: {
          title: 'Probability',
          minValue: 0
        },
        vAxis: {
          title: 'Nominee'
        },
	legend: { position: "none" },
      };

      var chart = new google.visualization.BarChart(document.getElementById('chart_div'));

    chart.draw(data, options);
    $('#txtQuery').val('');

}

function callPost()
{
    var query = $('#txtQuery').val();
    var spinner = new Spinner(opts).spin(target);

    $.ajax({
			url: '/queryModel',
			data: $('form').serialize(),
			type: 'POST',
		    success: function(response){
			response = jQuery.parseJSON(response)
			console.log(response)
			drawBasic(response['probabilities'], query)
			var feats = response['relevant_words'];
			$("ul").empty();
			var cList = $('ul.list-group')
			$.each(feats, function(i)
			       {
				   $('<li />', {class:"list-group-item", html: feats[i]}).appendTo(cList)
			       });
			},
			error: function(error){
				console.log(error);
			}
		});
}

$(function() {
    $("#txtQuery").keypress(function (e) {
        if(e.which == 13) {
	    callPost();
        }
    });
});




$(function(){
    $('button').click(function(){
	    $(this).val('');
	    callPost();
	});
});

$(function() {
    $('#example_table tr').click( function(){
	var text = $(this).find('td:first').html();
	$('#txtQuery').val(text);
	callPost()
});
});    
