import torch
import torch.nn as nn

from models import *
from config import *

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import networkx as nx

config.perception = "csqnet"
config.concept_type = "cone"
learner = SceneLearner(config)

def hex2rgb(h):
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb2hex(rgb):
    return '#{0:02x}{1:02x}{2:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def logit(x):
    x = x.clamp(EPS, 1-EPS)
    return torch.log(x/ (1 - x))


colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


gt_tree = nx.DiGraph()
gt_tree.add_nodes_from(["pot","body","container"])
gt_tree.add_edges_from([
    ("pot","body"),
    ("body","container"),

])

eval_data = []
#eval_data.append({"program":"exist(filter(scene(),{}))".format("foot"),"answer":"no"})
#eval_data.append({"program":"exist(filter(subtree(filter(scene(),{})),{}))".format("container","body"),"answer":"no"})
#eval_data.append({"program":"exist(filter(subtree(filter(scene(),{})),{}))".format("body","pot"),"answer":"no"})
for node in gt_tree.nodes:
    eval_data.append({"program":"exist(filter(scene(),{}))".format(node),"answer":"yes"})
for edge in gt_tree.edges:
    eval_data.append({"program":"exist(filter(subtree(filter(scene(),{})),{}))".format(edge[0],edge[1]),"answer":"yes"})
    eval_data.append({"program":"exist(filter(subtree(filter(scene(),{})),{}))".format(edge[1],edge[0]),"answer":"no"})

def get_prob(executor,feat,concept):
        pdf = []
        for predicate in executor.concept_vocab:
            pdf.append(torch.sigmoid(executor.entailment(feat,
                executor.get_concept_embedding(predicate) )))
        pdf = torch.cat(pdf, dim = 0)
        idx = executor.concept_vocab.index(concept)
        return pdf[idx]/ pdf.sum(dim = 0)

def build_label(feature, executor):
    default_label = "x"
    predicates = executor.concept_vocab
    prob = 0.0
    for predicate in predicates:
        pred_prob = get_prob(executor, feature, predicate)
        if pred_prob > prob:
            prob = pred_prob
            default_label = predicate
            #default_label = "{}_{:.2f}".format(predicate,float(pred_prob))

    default_color = [64./255,74./255,121./255,float(prob)]
    return default_label, default_color

def visualize_outputs(scores, features, connections,executor, kwargs):
    plt.cla()
    shapes = [score.shape[0] for score in scores]
    nodes = [];labels = [];colors = [];layouts = []
    # Initialize Scores
    width = 0.6; height = 1.0
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    for i in range(len(scores)):
        if len(scores[i]) == 1: xs = [0.0];
        else: xs = torch.linspace(-1,1,len(scores[i])) * (width ** i)
        for j in range(len(scores[i])):
            nodes.append(sum(shapes[:i]) + j)
            
            if len(features[i].shape) == 3:
                label,c = build_label(features[i][:,j], executor)
            else:label,c = build_label(features[i][j], executor)
            #c[0] = float(torch.linspace(0,1,len(scores))[i])
            c[-1] = min(float(scores[i][j]),c[-1])
            c[-1] = float(scores[i][j])
            label = "{}_{:.2f}".format(label,c[-1])
            labels.append(label);colors.append(c)
            # layout the locations
            layouts.append([xs[j],i * height])
            plt.scatter(xs[j],i * height,color = c, linewidths=9)
            plt.text(xs[j], i * height - 0.01, label)

    for n in range(len(connections)):
        connection = connections[n].permute(1,0)

        for i in range(len(connection)):
            for j in range(len(connection[i])):                
                u = i + sum(shapes[:n])
                v = j + sum(shapes[:n+1])
       
                plt.plot(
                    (layouts[u][0],layouts[v][0]),
                    (layouts[u][1],layouts[v][1]), alpha = float(connection[i][j].detach()), color = "black")

    return 


def load_scene(scene, k): 
    scores = scene["scores"]; features = scene["features"]; connections = scene["connections"]
    return [score[k] for score in scores], [feature[k] for feature in features], \
        [connection[k] for connection in connections[1:]]

# Set Level Grounding 
EPS = 1e-5

TEST = 1

config.hierarchy_construct = (3,2,1)

config.temperature = 2.55
model = SceneLearner(config)
struct = config.hierarchy_construct

if TEST:
    config.hierarchy_construct = (4,)
    struct = config.hierarchy_construct
    test_features = [
        Variable(torch.randn(s,100),requires_grad = True) for s in struct
    ]
    
    test_features.reverse ()

    test_scores = [
        torch.tensor([1,1,1,1]),
    ]
    test_scores.reverse()

    test_connections = []
    test_connections.reverse()
   
    eval_data = []
    for node in ["plant"]:
        eval_data.append({"program":"exist(filter(scene(),{}))".format(node),"answer":"yes"})
        eval_data.append({"program":"count(filter(scene(),{}))".format(node),"answer":"1"})
else:
    test_features = [
        Variable(torch.randn(s,100),requires_grad = True) for s in struct
    ]
    
base_features = Variable(torch.randn([1,3,100]), requires_grad=True)

for node in model.executor.concept_vocab:
    answer = "yes" if node in gt_tree.nodes else "no"

test_tree = nx.DiGraph()
test_tree.add_node("root")
test_tree.add_node("pot")
test_tree.add_node("body")
test_tree.add_node("container")
test_tree.add_node("plant")

test_tree.add_edge("root","pot")
test_tree.add_edge("pot","body")
test_tree.add_edge("body","container")
test_tree.add_edge("body","plant")
"""
抱歉 之前家里出了些事情 一直没能调整好自己的状态； 导致一直没有作出什么progress。
我在doc里记录了目前跑出来的一些结果 以及一些改进的propsal 不知道这样是否合理

多谢理解🙏
"""

def gen_full_grounding(test_tree, mode = "full"):
    test_dataset = {}
    nodes = [];depths = []
    sons = {}
    def dfs(node, depth = 0):
        nodes.append(node)
        depths.append(depth)
        depth += 1
        for edge in test_tree.edges:
            if edge[0] == node:
                if node not in sons:sons[node] = [edge[1]]     
                else:sons[node].append(edge[1]) 
                dfs(edge[1], depth = depth)
    dfs("root",0)
    D = max(depths)-1
    for d in range(D+1,0,-1):
        if mode == "full":
            test_data = []
            available_nodes = []
            top_nodes = []
            for i,node in enumerate(nodes):
                if depths[i] >= d:available_nodes.append(node)
                if depths[i] == d: top_nodes.append(node)

            # existence data
            for exist_node in available_nodes:
                test_data.append(
                    {"program:":"exist(filter(scene(),{}))".format(exist_node),
                    "answer":"yes"})
            # hierarchy data
            for top_node in top_nodes:
                if top_node in sons:
                    for son in sons[top_node]:
                        test_data.append(
                        {"program":"exist(filter(subtree(filter(scene(),{})),{}))".format(top_node, son),
                         "answer":"yes"})

            # count
            for node in available_nodes:
                answer = "yes" if node in available_nodes else "no"
                test_data.append(
                    {"program:":"count(filter(scene(),{}))".format(node),
                    "answer":"1","type":"counting",
                    "question":"count(filter(scene(),{}))".format(node)})

            # no count
            for node in model.executor.concept_vocab:
                answer = "yes" if node in available_nodes else "no"
                test_data.append(
                    {"program:":"exist(filter(scene(),{}))".format(node),
                    "answer":answer})

            test_dataset[str(D - d + 2)] = test_data
        if mode == "sample":
            test_dataset[str(D - d + 2)] = []
    test_tree.remove_edge("root","pot")
    test_tree.remove_node("root")
    return test_dataset, D+1

#for q in eval_data:print(q["program"],":",q["answer"])

test_datasets, scene_depth = gen_full_grounding(test_tree)
#print(test_datasets)

def grounding(input_features,model, data, epochs ,phase):
    optim = torch.optim.Adam([{'params': model.parameters()},
                            {'params':base_features},
                            {"params":input_features}], lr = 1e-2)
    
    for epoch in range(epochs):
        loss = 0
        scene = model.build_scene(input_features)
        test_scores, test_features, test_connections = load_scene(scene,0)

        kwargs = {"features":test_features, "end":test_scores, "connections":test_connections}
        plt.figure("Comparison",figsize=(14,6))
        plt.subplot(121)
        plt.cla()
        pos = nx.layout.spring_layout(test_tree)
        nx.draw_networkx(test_tree,pos)
        plt.subplot(122)
        visualize_outputs(test_scores, test_features, test_connections, model.executor, kwargs)
        plt.pause(0.001)

        losses = []
        for qa_pair in data:
            q = qa_pair[list(qa_pair.keys())[0]]
            answer = qa_pair["answer"]
            q = model.executor.parse(q)
            o = model.executor(q, **kwargs)

            if answer in ["True","False"]:answer = {"True":"yes,","False":"no"}[answer]
            if answer in ["1","2","3","4","5"]:answer = num2word(int(answer))
        
            if answer in numbers:
                int_num = torch.tensor(numbers.index(answer)).float()
                loss += + F.mse_loss(int_num ,o["end"]) 
                losses.append(logit(F.mse_loss(int_num,o["end"]).detach()))
            if answer in yes_or_no:
                if answer == "yes":
                    loss -= torch.log(torch.sigmoid(o["end"]))

                    losses.append(logit(torch.sigmoid(o["end"])).detach())
                else:
                    loss -= torch.log(1 - torch.sigmoid(o["end"]))
                    losses.append(logit(1 - torch.sigmoid(o["end"])).detach())

        
        sys.stdout.write("\rphase{}, epoch:{} loss:{}".format(phase,epoch,float(loss.detach())))

        optim.zero_grad()
        loss.backward()
        optim.step()
    for i,q in enumerate(data):
        output = "\r{}:{} ->{}\n".format(q[list(q.keys())[0]],q["answer"],str(float(torch.sigmoid(losses[i]))))
        sys.stdout.write(output)

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

def freeze_hierarchy(model, depth):
    size = len(model.scene_builder)
    for i in range(1,size+1):
        if i <= depth:unfreeze_parameters(model.hierarhcy_maps[i-1])
        else:freeze_parameters(model.scene_builder)
    model.executor.effective_level = depth

test_features = Variable(torch.randn(1,3,100),requires_grad = True)

for phase in range(1, scene_depth + 1):
    print("Start on phase {}:".format(phase),len(test_datasets[str(phase)]))
    freeze_hierarchy(model,depth = phase)
    grounding(test_features,model,test_datasets[str(phase)], 400, phase)
    print("Done in phase {}!".format(phase))

"""
for epoch in range(100000):
    loss = 0
    scene = model.build_scene(base_features)
    scores,features,connections = load_scene(scene, 0)

    kwargs = {"features":features,
                  "end":scores,
                 "connections":connections}

    # Use the Test Tree 
    if TEST:
        kwargs = {"features":test_features,
                  "end":test_scores,
                  "connections":test_connections}
        visualize_outputs(test_scores, test_features, test_connections, model.executor, kwargs)
    else:
        visualize_outputs(scores, features, connections, model.executor, kwargs)
    plt.pause(0.001)
    losses = []
    for qa_pair in eval_data:
        q = qa_pair["program"]
        answer = qa_pair["answer"]
        q = model.executor.parse(q)
        o = model.executor(q, **kwargs)

        if answer in ["True","False"]:answer = {"True":"yes,","False":"no"}[answer]
        if answer in ["1","2","3","4","5"]:answer = num2word(int(answer))
        
        if answer in numbers:
            int_num = torch.tensor(numbers.index(answer)).float()
            loss += + F.mse_loss(int_num ,o["end"])
            losses.append(logit(F.mse_loss(int_num,o["end"]).detach()))
        if answer in yes_or_no:
            if answer == "yes":
                loss -= torch.log(torch.sigmoid(o["end"]))
                losses.append(logit(torch.sigmoid(o["end"])).detach())
            else:
                loss -= torch.log(1 - torch.sigmoid(o["end"]))
                losses.append(logit(1 - torch.sigmoid(o["end"])).detach())

    sys.stdout.write("\r")
    for i,q in enumerate(eval_data):
        output = "\r{}:{} ->{}\n".format(q["program"],q["answer"],str(float(torch.sigmoid(losses[i]))))
        sys.stdout.write(output)
    sys.stdout.write("epoch:{} loss:{}".format(epoch,float(loss.detach())))

    optim.zero_grad()
    loss.backward()
    optim.step()

#for s in o["end"]:print(np.array((torch.sigmoid(s) + 0.5).int()))
"""