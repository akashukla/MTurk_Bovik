function orderGScores(golden_mu, golden_scores) {
    var gmeans_ordered = JSON.parse(JSON.stringify(golden_mu));
    var gscores_ordered = JSON.parse(JSON.stringify(golden_scores));
    var gmeans_i = [];
    var gmeans_sort = [];

    for (var i in golden_mu) {
        gmeans_sort.push([golden_mu[i], i]);
    }
    gmeans_sort.sort(function(left, right) {
        return left[0] < right[0] ? -1 : 1;
    });
    for (var j in gmeans_sort) {
        gmeans_ordered.push(gmeans_sort[j][0]);
        gmeans_i.push(gmeans_sort[j][1]);
    }

    for (var k = 0; k < golden_mu.length; k++) {
        gscores_ordered.push(golden_scores[gmeans_i[k]]);                
    }

    return {
        gmeans_ordered: gmeans_ordered,
        gscores_ordered: gscores_ordered
    };

}

var golden_mu = [3,2,1,5,4];
var golden_scores = [7,8,6,10,9];
var eval = orderGScores(golden_mu, golden_scores);
console.log(eval.gmeans_ordered.splice(golden_mu.length,golden_mu.length));
console.log(eval.gscores_ordered.splice(golden_mu.length,golden_mu.length));


