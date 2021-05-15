angular.module("Explorer").component("questionDetail", {
    templateUrl: "/GrailQA/explore/questionDetails.html",
    controller: function () {
        this.mp = model_predictions;
        this.preview_preds = function () {
            // return [0]
            return model_predictions[this.model][this.question.qid]
        }
        this.len = Object.keys(model_predictions).length;


        this.genGraphs = function () {
            var qid = this.question.qid;
            var query = this.question.graph_query;
            $("#expanded_" + this.question.qid).on("shown.bs.modal", function () {
                console.log("Shown");
                generate_graph(qid, query);
            });
            $("#expanded_" + this.question.qid).on("hide.bs.modal", function () {
                console.log("Hiding");
                clear_graph(qid);
            });
        }

        this.color_correct = function (event) {
            var className = ""
            if (event.type === "mouseenter") {
                var answer_args = this.question.answer.map(function (ans) { return ans.answer_argument });
                if (answer_args.includes(event.target.innerText)) { className = "correct" }
                else { className = "incorrect" }
            }
            event.target.className = className;
        }
    },
    bindings: {
        question: '=',
        model: '=',
    }
});
