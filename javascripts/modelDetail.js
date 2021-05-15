angular.module("Explorer").component("modelDetail", {
    templateUrl: "/GrailQA/explore/modelDetails.html",
    bindings: {
        model: '=',
        models: '=',
        qid: '=',
    },
    controller: function () {
        this.model_info = function () {
            return model_predictions[this.model][this.qid];
        }
    }
});
