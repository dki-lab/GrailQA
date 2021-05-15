// Borrowed from https://bost.ocks.org/mike/shuffle/
function shuffle(array) {
    var m = array.length, t, i;

    // While there remain elements to shuffle…
    while (m) {
        // Pick a remaining element…
        i = Math.floor(Math.random() * m--);

        // And swap it with the current element.
        t = array[m];
        array[m] = array[i];
        array[i] = t;
    }

    return array;
}

angular.module("Explorer", []).controller("ExplorerController", function ($scope) {
    this.total_data = shuffle(data);

    this.func_filter = '';
    this.domain_filter = '';
    this.complexity = { from: 1, to: 4 };
    this.generalization_filter = '';
    this.correct_filter = '';
    this.page = 1;
    this.max_qs = 50;

    this.model = Object.keys(model_predictions)[0];
    this.models = model_predictions;

    this.em = function () {
        var model_ans = model_predictions[this.model]
        var current_data = this.filtered_data()

        var correct = 0

        current_data.forEach(function (q) {
            actual = q["s_expression"]
            predicted = model_ans[q["qid"]] ? model_ans[q["qid"]]["logical_form"] : undefined;

            correct = predicted === actual ? correct + 1 : correct;
        })

        var em = correct / current_data.length;
        return (em * 100).toFixed(3);
    }

    this.f1 = function () {
        var model_ans = model_predictions[this.model]
        var current_data = this.filtered_data()

        var f1_sum = 0
        current_data.forEach(function (q) {
            var q_f1 = 0
            var actual = new Set(q.answer
                .map(function (el) { return el.answer_argument }));
            var predictions = undefined;
            if (model_ans[q.qid]) {
                if (model_ans[q.qid].answer.length > 0) {
                    predictions = new Set(model_ans[q.qid].answer);
                }
            }
            if (predictions) {
                intersection = new Set(Array.from(actual).filter(function (x) { return predictions.has(x) }));
                recall = intersection.size / actual.size;
                precision = intersection.size / predictions.size
                q_f1 = 2 / (1 / recall + 1 / precision);
            }
            f1_sum += q_f1;
        });

        f1_sum = f1_sum / current_data.length;
        return (f1_sum * 100).toFixed(3);
    }

    this.total = this.total_data.length;

    this.filtered_data = function () {
        var temp = this.total_data;

        if (this.func_filter && this.func_filter.length !== 0) {
            const args_mapper = {
                "comparative": ['<=', '<', '>', '>='],
                "superlative": ['argmin', 'argmax'],
                "count": ["count"],
                "none": ["none"]
            };
            const arg_set = args_mapper[this.func_filter];

            temp = temp.filter(function (el) { return arg_set.includes(el.function) });
        }
        if (this.domain_filter && this.domain_filter.length !== 0) {
            temp = temp.filter(function (el) { return el.domains.includes(this.domain_filter); }, this);
        }
        temp = temp.filter(function (el) { return el.num_edge <= this.complexity.to && el.num_edge >= this.complexity.from; }, this);
        if (this.generalization_filter && this.generalization_filter.length !== 0) {
            temp = temp.filter(function (el) { return el.level === this.generalization_filter; }, this);
        }
        if (this.correct_filter && this.correct_filter.length !== 0) {
            predictions = model_predictions[this.model]
            if (this.correct_filter === 'correct') {
                temp = temp.filter(function (el) { return Object.hasOwnProperty.call(predictions, el.qid) && el.s_expression === predictions[el.qid].logical_form }, this);
            } else {
                temp = temp.filter(function (el) { return Object.hasOwnProperty.call(predictions, el.qid) && el.s_expression !== predictions[el.qid].logical_form }, this);
            }
        }

        return temp;
    }

    this.display_data = function () {
        var filtered = this.filtered_data();

        // console.log("Finished Filtering");
        this.total = filtered.length;
        return filtered.slice(this.max_qs * (this.page - 1), (this.page * this.max_qs))
    }


    this.pageNums = function () {
        var all_page_nums = []
        for (var i = 0; i < Math.ceil(this.total / this.max_qs); i++) {
            all_page_nums.push(i + 1);
        }

        var show_page_nums = []
        const halfway = Math.floor(5 / 2);
        if (this.page <= halfway) {
            show_page_nums = all_page_nums.slice(0, 5);
        } else if (this.page >= all_page_nums.length - halfway) {
            show_page_nums = all_page_nums.slice(all_page_nums.length - 5);
        } else {
            show_page_nums = all_page_nums.slice(this.page - 2, this.page + 2)
        }

        return [show_page_nums, all_page_nums]
    }

    function setup_slider(complexity) {
        var min_handle = $("#min-handle");
        var max_handle = $("#max-handle");
        $("#complexity-slider").slider({
            range: true,
            min: 1,
            max: 4,
            step: 1,
            values: [1, 4],
            create: function () {
                min_handle.text(1)
                max_handle.text(4)
            },
            slide: function (event, ui) {
                $scope.$apply(function () {
                    min_handle.text(ui.values[0]);
                    complexity.from = ui.values[0];
                    max_handle.text(ui.values[1]);
                    complexity.to = ui.values[1];
                });
            }
        });
    }
    $(setup_slider(this.complexity));

    this.reset = function () {
        this.total_data = shuffle(data)
        this.func_filter = '';
        this.domain_filter = '';
        this.complexity = { from: 1, to: 4 };
        this.generalization_filter = '';
        $(setup_slider(this.complexity))
    }
});
