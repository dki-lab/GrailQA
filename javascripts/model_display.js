$("#model").on('change', function (event) {
    updateModelDiv(event.target.value);
});

function updateModelDiv(model_name) {
    var model_div = $('#model_info');
    model_div.innerHTML = '';
    createModelElement(model_name).forEach(function (block) {
        model_div.append(block);
    });
}

function createModelElement(model_name) {
    var model_info = predictions[model_name]
    var logi_div = document.createElement("div");
    var logi_form = document.createElement("p");
    logi_form.innerText = "Logical Form: ";
    var code = document.createElement("pre");
    code.innerText = model_info.logical_form;
    logi_form.appendChild(code);
    logi_div.appendChild(logi_form);
    var answer_div = document.createElement("div");
    var answer_header = document.createElement("p");
    answer_header.innerText = model_info.answer.length > 1 ? "Answers" : "Answer";
    answer_div.appendChild(answer_header)
    var list_wrapper = document.createElement("ul");
    list_wrapper.className = "list-group list-group-flush";
    model_info.answer.forEach(function (answer) {
        var ans = document.createElement("li");
        var ans_text = document.createElement("samp")
        ans_text.innerText = answer;
        ans.className = "list-group-item"
        ans.appendChild(ans_text)
        list_wrapper.appendChild(ans);
    });
    answer_div.appendChild(list_wrapper);
    logi_div.appendChild(answer_div);

    return [logi_div, answer_div];
}

updateModelDiv($('#model_selector')[0].value);