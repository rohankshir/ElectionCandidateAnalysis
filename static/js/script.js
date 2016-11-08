google.charts.load('current', {packages: ['corechart', 'bar']});
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

    $.ajax({
			url: '/queryModel',
			data: $('form').serialize(),
			type: 'POST',
		    success: function(response){
			response = jQuery.parseJSON(response)
			console.log(response)
			drawBasic(response['probabilities'], query)
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
