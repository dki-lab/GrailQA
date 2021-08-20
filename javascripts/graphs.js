var simulation = 0

function clear_graph(qid) {
    const svg_id = "graph_" + qid;
    document.getElementById(svg_id).innerHTML = '';
    simulation.stop()
}

function generate_graph(qid, graph) {
    graph.edges.forEach(function (_, i) {
        graph.edges[i].source = graph.edges[i].start
        graph.edges[i].target = graph.edges[i].end
    });

    const svg_id = "graph_" + qid;
    document.getElementById(svg_id).innerHTML = '';
    var width = document.getElementById(svg_id).width.baseVal.value;
    var height = document.getElementById(svg_id).height.baseVal.value;
    var svg = d3.select('#' + svg_id)
        // .attr('viewBox', [-width / 2, -height / 2, width, height])
        .attr('width', width)
        .attr('height', height);

    simulation = d3.forceSimulation(graph.nodes)
        .force("link", d3.forceLink(graph.edges).id(function (d) {
            return d.nid;
        }))
        .force("charge", d3.forceManyBody().strength(-100))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide(function () {
            return 65;
        }));

    var drag = function (simulation) {
        function dragstarted(event, d) {
            if (!event.active)
                simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active)
                simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }

    const nodeRadius = 0.1 * Math.sqrt(width * height) / graph.nodes.length;

    // Per-type markers, as they don't inherit styles.
    svg.append("defs").append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 38)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("svg:path")
        .attr("fill", "#888")
        .attr("d", 'M0,-5L10,0L0,5');

    const link = svg.append("g")
        .attr("fill", "none")
        .attr("stroke-width", 1.5)
        .selectAll("path")
        .data(graph.edges)
        .join("path")
        .attr("stroke", "#888")
        .attr("marker-end", function () { return "url(#arrowhead)"; });

    const edgepaths = svg.selectAll(".edgepath")
        .data(graph.edges)
        .enter()
        .append('path')
        .attr('class', 'edgepath')
        .attr('fill-opacity', 0)
        .attr('stroke-opacity', 0)
        .attr('id', function (d, i) { return 'edgepath' + i })
        // .attr("marker-end", "url(#arrowhead)")
        .style("pointer-events", "none");

    const edgelabels = svg.selectAll(".edgelabel")
        .data(graph.edges)
        .enter()
        .append('text')
        .attr('class', 'edgelabel')
        .attr('id', function (d, i) { return 'edgelabel' + i })
        .attr('font-size', 10)
        .attr('fill', '#000')
        .style("pointer-events", "none");

    edgelabels.append('textPath')
        .attr('xlink:href', function (d, i) { return '#edgepath' + i })
        .style("text-anchor", "middle")
        .style("pointer-events", "none")
        .attr("startOffset", "50%")
        .text(function (d) { return d.friendly_name });

    const node = svg.append("g")
        .attr("fill", "currentColor")
        .attr("stroke-linecap", "round")
        .attr("stroke-linejoin", "round")
        .selectAll("g")
        .data(graph.nodes)
        .join("g")
        .call(drag(simulation));

    node.append("circle")
        .attr("stroke", "white")
        .attr("stroke-width", 1.5)
        .attr("r", nodeRadius)
        .attr('fill', function (d) { return d.nid != 0 ? "#d8ab1f" : "#7e7e7e"; });

    node.append("text")
        // .attr("stroke", "white")
        .attr('x', nodeRadius + 1)
        .attr('y', 3)
        .text(function (d) {
            return d.friendly_name + ":" + d.node_type;
        })

    node.on('dblclick', function (e, d) {
        return console.log(graph.nodes[d.index]);
    })

    var linkArc = function (d) {
        return 'M' + d.source.x
            + ',' + d.source.y
            + 'A0,0 0 0,1 ' + d.target.x
            + ',' + d.target.y;
    }

    simulation.on("tick", function () {
        node.attr("transform", function (d) {
            d.x = Math.max(nodeRadius, Math.min(width - nodeRadius, d.x));
            d.y = Math.max(nodeRadius, Math.min(height - nodeRadius, d.y));
            return 'translate(' + d.x + ',' + d.y + ')';
        });
        link.attr("d", linkArc);

        edgepaths.attr('d', function (d) {
            return 'M ' + d.source.x + ' ' + d.source.y + ' L ' + d.target.x + ' ' + d.target.y;
        });

        edgelabels.attr('transform', function (d) {
            if (d.target.x < d.source.x) {
                var bbox = this.getBBox();

                // console.log(bbox)

                rx = bbox.x + bbox.width / 2;
                ry = bbox.y + bbox.height / 2;
                return 'rotate(180 ' + rx + ' ' + ry + ')';
            }
            else {
                return 'rotate(0)';
            }
        })
    });
}

// graph.edges.forEach(function (_, i) {
//     graph.edges[i].source = graph.edges[i].start
//     graph.edges[i].target = graph.edges[i].end
// });


// const svg_id = "graph_" + qid;
// document.getElementById(svg_id).innerHTML = '';
// var svg = d3.select('#' + svg_id),
//     width = +document.getElementById(svg_id).width.baseVal.value,
//     height = +document.getElementById(svg_id).height.baseVal.value;


// const nodeRadius = 0.1 * Math.sqrt(width * height) / graph.nodes.length;

// svg.append('defs').append('marker')
//     .attr('id', 'arrowhead')
//     .attr('viewBox', '-0 -5 10 10')
//     .attr('refX', 13)
//     .attr('refY', 0)
//     .attr('orient', 'auto')
//     .attr('id', 'arrowhead')
//     .attr('markerWidth', 13)
//     .attr('xoverflow', 'visible')
//     .append('svg:path')
//     .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
//     .attr('fill', '#999')
//     .style('stroke', 'none');

// var simulation = d3.forceSimulation()
//     .force("link", d3.forceLink().id(function (d) { return d.nid; }).distance(5 * nodeRadius).strength(1))
//     .force("charge", d3.forceManyBody())
//     .force("center", d3.forceCenter(width / 2, height / 2));

// link = svg.selectAll(".link")
//     .data(graph.edges)
//     .enter()
//     .append("line")
//     .attr("stroke", "#888")
//     .attr("class", "link")
//     .attr('marker-end', 'url(#arrowhead)')

// link.append("title")
//     .text(function (d) { return d.friendly_name; });

// edgepaths = svg.selectAll(".edgepath")
//     .data(graph.edges)
//     .enter()
//     .append('path')
//     .attr('class', 'edgepath')
//     .attr('fill-opacity', 0)
//     .attr('stroke-opacity', 0)
//     .attr('id', function (d, i) { return 'edgepath' + i })
//     .style("pointer-events", "none");

// edgelabels = svg.selectAll(".edgelabel")
//     .data(graph.edges)
//     .enter()
//     .append('text')
//     .attr('class', 'edgelabel')
//     .attr('id', function (d, i) { return 'edgelabel' + i })
//     .attr('font-size', 10)
//     .attr('fill', '#000')
//     .style("pointer-events", "none");

// edgelabels.append('textPath')
//     .attr('xlink:href', function (d, i) { return '#edgepath' + i })
//     .style("text-anchor", "middle")
//     .style("pointer-events", "none")
//     .attr("startOffset", "50%")
//     .text(function (d) { return d.friendly_name });

// node = svg.selectAll(".node")
//     .data(graph.nodes)
//     .enter()
//     .append("g")
//     .attr("class", "node")
//     .call(d3.drag()
//         .on("start", dragstarted)
//         .on("drag", dragged)
//         //.on("end", dragended)
//     );

// node.append("circle")
//     .attr("r", 5)
//     .style("fill", function (d) { return d.nid != 0 ? "#d8ab1f" : "#7e7e7e"; })

// node.append("title")
//     .text(function (d) { return d.nid; });

// node.append("text")
//     .attr("dy", -3)
//     .attr('x', nodeRadius + 1)
//     .attr('y', 3)
//     .text(function (d) { return d.friendly_name + ":" + d.node_type; });

// simulation
//     .nodes(graph.nodes)
//     .on("tick", ticked);

// simulation.force("link")
//     .links(graph.edges);


// function ticked() {
//     link
//         .attr("x1", function (d) { return d.source.x; })
//         .attr("y1", function (d) { return d.source.y; })
//         .attr("x2", function (d) { return d.target.x; })
//         .attr("y2", function (d) { return d.target.y; });

//     node
//         .attr("transform", function (d) { return "translate(" + d.x + ", " + d.y + ")"; });

//     edgepaths.attr('d', function (d) {
//         return 'M ' + d.source.x + ' ' + d.source.y + ' L ' + d.target.x + ' ' + d.target.y;
//     });

//     edgelabels.attr('transform', function (d) {
//         if (d.target.x < d.source.x) {
//             var bbox = this.getBBox();

//             rx = bbox.x + bbox.width / 2;
//             ry = bbox.y + bbox.height / 2;
//             return 'rotate(180 ' + rx + ' ' + ry + ')';
//         }
//         else {
//             return 'rotate(0)';
//         }
//     });
// }

// function dragstarted(event, d) {
//     if (!event.active) simulation.alphaTarget(0.3).restart()
//     graph.nodes[d].fx = graph.nodes[d].x;
//     graph.nodes[d].fy = graph.nodes[d].y;
//     console.log(graph.nodes[d].fx + ", " + graph.nodes[d].fy)
// }

// function dragged(event, d) {
//     graph.nodes[d].fx = event.x;
//     graph.nodes[d].fy = event.y;
//     console.log(graph.nodes[d].fx + ", " + graph.nodes[d].fy)
// }