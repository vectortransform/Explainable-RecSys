$(document).ready(function(){
    $("#rankbutton").on("click", function(){
        var user_id = $("#userid").val();
        var item_ids = $("#res").text();

        $.ajax({
            url: '/ranking',
            dataType: 'JSON',
            type: 'POST',
            data: JSON.stringify({
                uid: user_id,
                iids: item_ids
            }),
            contentType: 'application/json;charset=UTF-8',
            success: function(response){
                console.log(response);
                var rating = response['rating'];
                var ranked_ids = response['item_ids'];
                var infer_time = response['infertime'];

                $("#description").text("Predicted ratings: "+infer_time+" seconds for 100 inputs");
                for(var i = 0; i < 10; i++){
                    for(var j = 0; j < 10; j++){
                        var idx = 10*i+j;
                        $("#iid_"+idx.toString()).text("#"+ranked_ids[idx]);
                        $("#iid_"+idx.toString()).click(function() {
                            var iid = $(this).text().substring(1);
                            var uid = $("#userid").val();
                            getreviews(uid, iid);
                            $(window).scrollTop(0);
                        });
                        $("#irating_"+idx.toString()).text("Rating: "+rating[idx].toFixed(2)+"; Rank: "+(idx+1).toString()+"/100");
                        $("#irating_"+idx.toString()).css("font-weight","Bold");
                    }
                }

                var des_meta=response['des_meta'];
                var title_meta=response['title_meta'];
                var price_meta=response['price_meta'];
                var imurl_meta=response['imurl_meta'];
                var categ_meta=response['categ_meta'];

                for(var i = 0; i < 10; i++){
                    var meta_area = $("#ifig_"+i);
                    meta_area.append($("<br/>"));
                    meta_area.prepend('<img src='+imurl_meta[i]+' alt='+title_meta[i]+' width="350" />');
                    meta_area.append($("<br/>"));
                    meta_area.append(title_meta[i]);
                    meta_area.append($("<br/>"));
                    meta_area.append("Price: $"+price_meta[i]);
                }
            }
        });
    });

    $("#back_btn").on("click", function(){
        $("#grids").show();
        $("#review_grids").css('display','none');
        $("#toprev_title").css('display','none');
        $("#back_btn").css('display','none');
    });
};


function getreviews(uid, iid)
{
    $.ajax({
        url: '/predictreview',
        dataType: 'JSON',
        type: 'POST',
        data: JSON.stringify({
            uid: uid,
            iid: iid
        }),
        contentType: 'application/json;charset=UTF-8',
        success: function(response){
            $("#grids").css('display','none');
            $("#description").css('display','none');
            $("#rev_iid").text(iid);
            $("#toprev_title").show();
            $("#review_grids").show();
            $("#back_btn").show();
            var toprevs = response['toprevs'];
            var otherrevs = response['otherrevs'];

            $("#item_info").empty();
            var des_meta=response['des_meta'];
            var title_meta=response['title_meta'];
            var price_meta=response['price_meta'];
            var imurl_meta=response['imurl_meta'];
            var categ_meta=response['categ_meta'];

            $("#item_info").prepend('<img style="display: block; margin:auto" src='+imurl_meta+' alt='+title_meta+' height="350" />');
            $("#item_info").append($("<br/>"));
            $("#item_info").append('<p style="font-size:30px; text-align: center">' + title_meta + '</p>');

            if (des_meta=="") {
                $("#item_info").append('Description: None');
            } else {
                $("#item_info").append('Description: '+des_meta);
            }

            $("#item_info").append($("<br/>"));
            $("#item_info").append('Categories: '+categ_meta);
            $("#item_info").append($("<br/>"));
            $("#item_info").append('Price: $'+price_meta);

            $("#rev_type1").text('');
            var rev_rate_top = response['rev_rate_top'];
            for(var i = 0; i < toprevs.length; i++){
                var j=i+1;
                $("#rev_type1").append(j+": (");
                $("#rev_type1").append('<span style="font-size:20px; font-weight:Bold; color:red">' + rev_rate_top[i]) + '</span>';
                $("#rev_type1").append(" Star) "+toprevs[i]);
                $("#rev_type1").append($("<br/>"));
                $("#rev_type1").append($("<br/>"));
            }

            $("#rev_type3").text('');
            var rev_rate_other = response['rev_rate_other'];
            for(var i = 0; i < otherrevs.length; i++){
                var j=i+1;
                $("#rev_type3").append(j+": (");
                $("#rev_type3").append('<span style="font-size:20px; font-weight:Bold; color:red">' + rev_rate_other[i]) + '</span>';
                $("#rev_type3").append(" Star) "+otherrevs[i]);
                $("#rev_type3").append($("<br/>"));
                $("#rev_type3").append($("<br/>"));
            }
        }
    });
};
