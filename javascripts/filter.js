function filterThis(params) {
    var temp = data;
    if (params) {
        if (params.hasOwnProperty("func_filter")) {
            func_filter = params["func_filter"];
            if (func_filter && func_filter.length !== 0) {
                temp = data.filter(function (el) { return el.function === func_filter });
            }
        }
        if (params.hasOwnProperty("domain_filter")) {
            domain_filter = params["domain_filter"];
            if (domain_filter && domain_filter.length !== 0) {
                temp = temp.filter(function (el) { return el.domains.includes(domain_filter) });
            }
        }
        if (params.hasOwnProperty("complexity_filter_min")) {
            min_complexity = params["complexity_filter_min"];
            if (min_complexity && min_complexity.length !== 0) {
                temp = temp.filter(function (el) { return el.num_edge >= min_complexity });
            }
        }
        if (params.hasOwnProperty("complexity_filter_max")) {
            max_complexity = params["complexity_filter_max"];
            if (max_complexity && max_complexity.length !== 0) {
                temp = temp.filter(function (el) { return el.num_edge <= max_complexity });
            }
        }
        if (params.hasOwnProperty("generalization_filter")) {
            gen_filter = params["generalization_filter"];
            if (gen_filter && gen_filter.length !== 0) {
                temp = temp.filter(function (el) { return el.level === gen_filter });
            }
        }
    }

    var count_field = document.getElementById("count");
    count_field.innerText = "(" + temp.length + " of " + data.length + ")";

    var list_qs = document.getElementById("list-qs");
    list_qs.innerHTML = "";
    temp.forEach(function (element) {
        list_qs.appendChild(QAElement(element));
    });

    console.log("Finished Filtering");
}

function QAElement(element) {
    var qa_div = document.createElement("div");
    qa_div.className = "qa_wrapper";
    var row_div = document.createElement("div");
    row_div.className = "row";
    var col1_div = document.createElement("div");
    col1_div.className = "col-md-6";
    var qid_a = document.createElement("a");
    qid_a.href = element.qid + ".html";
    var qid = document.createElement("h6");
    qid.innerText = "ID: " + element.qid;
    qid_a.appendChild(qid);
    col1_div.appendChild(qid_a);
    var q = document.createElement("pre");
    q.className = "question";
    q.innerText = element.question;
    col1_div.appendChild(q);
    row_div.appendChild(col1_div);
    var col2_div = document.createElement("div");
    col2_div.className = "col-md-6";
    var dl_wrapper = document.createElement("dl");
    var dt_wrapper = document.createElement("dt");
    var header = document.createElement("i");
    header.innerText = "Answers: ";
    dt_wrapper.appendChild(header);
    dl_wrapper.appendChild(dt_wrapper);
    element.answer.slice(0, 5).forEach(function (answer) {
        var ans = document.createElement("dd");
        if (answer.answer_type === "Entity") {
            ans.innerText = answer.entity_name;
        } else {
            ans.innerText = answer.answer_argument;
        }
        // Indicate # not shown - Limit to 5
        dl_wrapper.appendChild(ans);
    });
    if (element.answer.length > 5) {
        var excess = document.createElement("dd");
        excess.innerText = "and " + element.answer.slice(5).length + " more..."
        excess.className = "text-muted"
        dl_wrapper.appendChild(excess)
    }
    col2_div.appendChild(dl_wrapper);
    row_div.appendChild(col2_div);
    qa_div.appendChild(row_div);

    return qa_div;
}

var filters = document.getElementById("filters");
filters.addEventListener("change", function (event) {
    const filter_labels = ["func_filter", "domain_filter", "complexity_filter_min", "complexity_filter_max", "generalization_filter"];
    var params = {};
    filter_labels.forEach(function (label) {
        element = document.getElementById(label);
        value = element.value;
        params[label] = value;
    });

    filterThis(params);
});

filterThis();
