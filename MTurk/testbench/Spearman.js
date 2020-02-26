function getAVG(a) {
    sum = 0;
    for(let i = 0; i < a.length; i++) {
        sum += a[i];
    }
    return sum/a.length;
}
// gets average of one column of a 2D array
function colAVG(a,idx) {
    sum = 0;
    for(let i = 0; i < a.length; i++) {
        sum += a[i][idx]
    }
    return sum/a.length
}

// takes care of duplicates within dataset and reorders it by ranking
function dup_order(x) {
    var a = x.slice()
    a.sort(function(i,j){return i[1]-j[1]})
    for(let i = 0; i < a.length; i++) {
        a[i][2] = i
    }

    let dup_num = false
    let dup_int = []
    let dup_list = []
    for(let i = 0; i < a.length; i++) {
        if(i == a.length-1) {
            if(dup_int.length != 0)
                dup_list.push(dup_int)
            break
        }
        if(dup_num == false) {
            if(a[i][1] == a[i+1][1]){
                dup_num = true
                dup_int.push(a[i][2])
                dup_int.push(a[i+1][2])
            }
        }
        else {
            if(a[i][1] == a[i+1][1]) {
                dup_int.push(a[i+1][2])
            }
            else {
                dup_list.push(dup_int)
                dup_int = []
                dup_num = false
            }
        }
    }
    a.sort(function(i,j){return i[1]-j[1]})
    var a_i = Array.from({ length: a.length}, (_,k) => k)
    for(let i = 0; i < dup_list.length; i++) {
        dup_avg = getAVG(dup_list[i])
        for (let j = 0; j < dup_list[i].length; j++) {
            a_i[dup_list[i][j]] = dup_avg
        }
    }

    for(let i = 0; i < a.length; i++) {
        a[i][2] = a_i[i]
    }

    a.sort(function(i,j){return i[0]-j[0]})
    return a
    
}

// creates data type that keeps track of [image ID, rating, ranking]
function tuple(a) {
    return Array.from({ length: a.length}, (_,k) => [k,a[k],-1])
}
// a and b must be arrays of the same length
function spearman(a,b) {
    var a_ord = dup_order(tuple(a))
    var b_ord = dup_order(tuple(b))

    var a_bar = colAVG(a_ord,2)
    var b_bar = colAVG(b_ord,2)
    
    var sum_a = 0
    var sum_b = 0
    var sum_c = 0
    for(let i = 0; i < a_ord.length; i++) {
        var a_diff = a_ord[i][2]-a_bar
        var b_diff = b_ord[i][2]-b_bar
        sum_a += a_diff * b_diff
        sum_b += a_diff * a_diff
        sum_c += b_diff * b_diff 
    }
    return ((sum_a)/Math.sqrt(sum_b*sum_c))
}

var a = [28,26,33,28,16,23,25,30]
var b = [42,21,43,23,22,25,16,22]

console.log(spearman(a,b))
