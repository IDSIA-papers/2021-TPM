module LearnPSDDasSPN

using LogicCircuits
using ProbabilisticCircuits
using DataFrames
using Parameters
using Clustering
using MappedArrays
using BenchmarkTools
using ProgressMeter
using Metis
using SparseArrays
using LightGraphs: add_edge!
using SimpleWeightedGraphs
using MetaGraphs
using LogicCircuits.LoadSave: get_vtree2id
using Metis: idx_t, ishermitian
import Base.Threads.@spawn

#using Random, ParallelKMeans
#using TikzPictures
#TikzPictures.standaloneWorkaround(true)
# n = num_examples(data)

STRUDEL_ITERATIONS = 100 #/ 100
FAST = false
LEARNVTREE = true
VERBOSE = true
δINT = 999999
MIN_INT = 1
MAX_INT = δINT + MIN_INT

@with_kw mutable struct mynode
    vtree::PlainVtree # = nothing
    db::DataFrame = DataFrame() # nothing #::DataFrame = DataFrame()
    label::String = "D" # Label
    position::String = ""
    parent::Int = -1
    literal::Int = 0
    visited::Bool = false
    theta::Float64 = -1.0
    left::Vector{Int64} = []
    right::Vector{Int64} = []
end

struct WeightedGraph
    nvtxs::idx_t
    xadj::Vector{idx_t}
    adjncy::Vector{idx_t}
    adjwgt::Vector{idx_t} # edge weights
    WeightedGraph(nvtxs, xadj, adjncy, adjwgt) = new(nvtxs, xadj, adjncy, adjwgt)
end

function my_graph(G::SparseMatrixCSC; check_hermitian=true)
    if check_hermitian
        ishermitian(G) || throw(ArgumentError("matrix must be Hermitian"))
    end
    N = size(G, 1)
    xadj = Vector{idx_t}(undef, N+1)
    xadj[1] = 1
    adjncy = Vector{idx_t}(undef, nnz(G))
    adjncy_i = 0
    adjwgt = Vector{idx_t}(undef, nnz(G))
    @inbounds for j in 1:N
        n_rows = 0
        for k in G.colptr[j] : (G.colptr[j+1] - 1)
            i = G.rowval[k]
            if i != j # don't include diagonal elements
                n_rows += 1
                adjncy_i += 1
                adjncy[adjncy_i] = i
                adjwgt[adjncy_i] = G[i, j]
            end
        end
        xadj[j+1] = xadj[j] + n_rows
    end
    resize!(adjncy, adjncy_i)
    resize!(adjwgt, adjncy_i)
    return WeightedGraph(idx_t(N), xadj, adjncy, adjwgt)
end

function my_partition(G::WeightedGraph, nparts::Integer)
    part = Vector{Metis.idx_t}(undef, G.nvtxs)
    edgecut = fill(idx_t(0), 1)
    Metis.METIS_PartGraphRecursive(G.nvtxs, idx_t(1), G.xadj, G.adjncy, C_NULL, C_NULL, G.adjwgt,
                                 idx_t(nparts), C_NULL, C_NULL, Metis.options, edgecut, part)
    return part
end

function clustering_can_be_done(unique_row::Int64, n_clusters::Int64, threshold::Int64)
    return unique_row > threshold && n_clusters > 1
end

function expand_psdd(nd::mynode,active::Int64,titles::Vector{String},ncl::Int64,thr::Int64)
    if LEARNVTREE
        l = typeof(nd.vtree.left) == PlainVtreeLeafNode ? [Int(variable(nd.vtree.left))] : [Int(n) for n in variables(nd.vtree.left)]
        r = typeof(nd.vtree.right) == PlainVtreeLeafNode ? [Int(variable(nd.vtree.right))] : [Int(n) for n in variables(nd.vtree.right)]
    else
        l = nd.left
        r = nd.right
    end
    dl = nd.db[:,find_names(names(nd.db),l,titles)]
    dr = nd.db[:,find_names(names(nd.db),r,titles)]
    if typeof(nd.vtree.left) == PlainVtreeInnerNode # Many vars on the left
        if clustering_can_be_done(nrow(unique(dl)),ncl,thr)
            return expand_many_left_multiclusters(nd::mynode,active::Int64,dl,dr) #return n_c left and right clusters
        else
            return expand_many_left_singlecluster(nd::mynode,active::Int64,dl,dr)
        end
    else
        if typeof(nd.vtree.right) == PlainVtreeInnerNode # Many variables on the right
            return expand_single_left_many_right(nd::mynode,active::Int64,dl,dr)
        else
            return expand_single_left_one_right(nd::mynode,active::Int64,dl,dr)
        end
    end
    #=if length(nd.left) > 1 # Many vars on the left
        if clustering_can_be_done(nrow(unique(dl)),ncl,thr)
            return expand_many_left_multiclusters(nd::mynode,active::Int64,dl,dr) #return n_c left and right clusters
        else
            return expand_many_left_singlecluster(nd::mynode,active::Int64,dl,dr)
        end
    else
        if length(nd.right) > 1 # Many variables on the right
            return expand_single_left_many_right(nd::mynode,active::Int64,dl,dr)
        else
            return expand_single_left_one_right(nd::mynode,active::Int64,dl,dr)
        end
    end=#
end

function expand_many_left_multiclusters(nd::mynode,active::Int64,d_left::DataFrame,d_right::DataFrame)
n_cl = 2
elements = []
indx_l = clustering4(d_left, n_cl) # do clustering
for c = 1:n_cl
    d_left_c = d_left[findall(x->x==c, indx_l),:]
    rename!(d_left_c,names(d_left))
    d_right_c = d_right[findall(x->x==c, indx_l),:]
    p = count(i->(i==c),indx_l)/length(indx_l)
    prime = mynode(vtree = nd.vtree.left, db = d_left_c, parent = active, position = "prime", theta = p)
    if typeof(nd.vtree.right) == PlainVtreeInnerNode # Many variables on the right
        sub = mynode( vtree = nd.vtree.right, db = d_right_c, parent = active, position = "sub")
    else # Single variable on the right
        var = Int64(variable(nd.vtree.right)) # variable as in the vtree
        nt = count(i->(i == true), d_right_c[:,1]) # number of true states in the right db
        if nt == size(d_right_c,1) || nt == 0
            lit = nt == 0 ? -var : var
            sub = mynode( vtree = nd.vtree.right, label = "L", visited = true, position = "sub", parent = active, literal = lit)
        else
            sub = mynode( vtree = nd.vtree.right, label = "T", visited = true, position = "sub", parent = active, literal = var, theta=nt/size(d_right_c,1))
        end
    end
    push!(elements,prime)
    push!(elements,sub)
end
return elements
end

function expand_many_left_singlecluster(nd::mynode,active::Int64,d_left::DataFrame,d_right::DataFrame)
prime = mynode(vtree = nd.vtree.left, db = d_left, parent = active, position = "prime", theta = 1.0)
if typeof(nd.vtree.right) == PlainVtreeInnerNode # Many variables on the right
    sub = mynode(vtree = nd.vtree.right, db = d_right, position = "sub", parent = active)
else # Single variable on the right
    var = Int64(variable(nd.vtree.right)) # variable as in the vtree
    nt = count(i->(i == true), d_right[:,1]) # number of true states in the right db
    if nt == size(d_right,1) || nt == 0
        lit = nt == 0 ? -var : var
        sub = mynode(vtree=nd.vtree.right,label="L",visited=true,position="sub",parent=active,literal=lit)
    else
        sub = mynode(vtree=nd.vtree.right,label="T",visited=true,position="sub",parent=active,literal=var,theta=nt/size(d_right,1))
    end
end
return [prime,sub]
end

function expand_single_left_many_right(nd::mynode,active::Int64,d_left::DataFrame,d_right::DataFrame)
    var = Int64(variable(nd.vtree.left)) # variable as in the vtree
    nt = count(i->(i == true), d_left[:,1]) # number of true states in the first (unique) col of db d
    if nt == size(nd.db,1) || nt == 0
        lit = nt == 0 ? -var : var
        prime = mynode(vtree = nd.vtree.left, label = "L", visited = true, position = "prime", parent = active, literal = lit, theta = 1.0)
        sub = mynode(vtree = nd.vtree.right, db = d_right, position = "sub", parent = active)
        elements = [prime,sub]
    else # some true and some false, Top with a weight
        prime1 = mynode(vtree=nd.vtree.left, label = "L", visited = true, position="prime", parent=active, literal=var, theta=nt/size(d_left,1))
        prime2 = mynode(vtree=nd.vtree.left, label = "L", visited = true, position="prime", parent=active, literal=-var, theta=1.0-nt/size(d_left,1))
        sub1 = mynode(vtree=nd.vtree.right, db=d_right[d_left[:,1],:], position="sub", parent=active)
        sub2 = mynode(vtree=nd.vtree.right, db=d_right[.!d_left[:,1],:], position="sub", parent=active)
        elements = [prime1,sub1,prime2,sub2]
    end
    return elements
end

function expand_single_left_one_right(nd::mynode,active::Int64,d_left::DataFrame,d_right::DataFrame)
var = Int16(variable(nd.vtree.left)) # variable as in the vtree
nt = count(i->(i == true), d_left[:,1]) # number of true states in the first (unique) col of db d
lit = nt == 0 ? -var : var
var2 = Int16(variable(nd.vtree.right))
nt2 = count(i->(i == true), d_right[:,1]) # number of true states in the first (unique) col of db d
lit2 = nt2 == 0 ? -var2 : var2
if nt == size(d_left,1) || nt == 0
    prime = mynode(vtree = nd.vtree.left, label = "L", visited = true, position = "prime", parent = active, literal = lit, theta = 1.0)
    if nt2 == size(d_right,1) || nt2 == 0
        sub = mynode(vtree = nd.vtree.right, label = "L", visited = true, position = "sub", parent = active, literal = lit2)
    else # Literal as prime, T with theta as a sub
        sub = mynode(vtree = nd.vtree.right, label = "T", visited = true, position = "sub", parent = active, literal = var2, theta = nt2/size(d_right,1))
    end
    return [prime,sub]
else
    prime1 = mynode(vtree = nd.vtree.left, label = "L", visited = true, position = "prime", parent = active, literal = var, theta = nt/size(d_left,1))
    prime2 = mynode(vtree = nd.vtree.left, label = "L", visited = true, position = "prime", parent = active, literal = -var, theta = 1.0-nt/size(d_left,1))
    d_right_1 = d_right[d_left[:,1],:]
    nt_1 = count(i->(i == true), d_right_1[:,1]) # number of true states in the first (unique) col of db d
    if nt_1 == size(d_right_1,1) || nt_1 == 0
        lit3 = nt_1 == 0 ? -var2 : var2
        sub1 = mynode(vtree = nd.vtree.right, label="L", visited = true, position = "sub", parent = active, literal = lit3)
    else
        sub1 = mynode(vtree = nd.vtree.right, label="T", visited = true, position = "sub", parent = active, literal = var2, theta = nt_1/size(d_right_1,1))
    end
    d_right_2 = d_right[.!d_left[:,1],:]
    nt_2 = count(i->(i == true), d_right_2[:,1]) # number of true states in the first (unique) col of db d
    if nt_2 == size(d_right_2,1) || nt_2 == 0
        lit4 = nt_2 == 0 ? -var2 : var2
        sub2 = mynode(vtree=nd.vtree.right, label="L", visited = true, position = "sub", parent = active, literal = lit4)
    else
        sub2 = mynode(vtree=nd.vtree.right, label="T", visited = true, position = "sub", parent = active, literal = var2, theta = nt_2/size(d_right_2,1))
    end
    return [prime1,sub1,prime2,sub2]
end
end

function experiment(db_name::String,nc::Int64,thresh::Int64,test_size::Int64=-1) # DEBUG: typing
    train_x, valid_x, test_x = twenty_datasets(db_name)
    if test_size > 0
        test_x = test_x[1:test_size,:]
    end
    test_size = size(test_x,1)
    train_size = size(train_x,1)
    n_features = size(test_x,2)
    unique1 = size(unique(train_x),1)
    unique2 = size(unique(test_x),1)
    open("$db_name-$nc-$thresh-$test_size.results", "w") do myfile
    if VERBOSE
        println("[$db_name][DB] n_c=$nc, threshold=$thresh, |X|=$n_features, STRUDEL_ITERATIONS=$STRUDEL_ITERATIONS, dim_train/test=$train_size($unique1)/$test_size($unique2)")
    end
    write(myfile,"[$db_name][DB] n_c=$nc, threshold=$thresh, |X|=$n_features, dim_train/test=$train_size($unique1)/$test_size($unique2)\n")
    #@time
    pc_cl, vtree_cl = learn_chow_liu_tree_circuit(train_x)
    if VERBOSE
        println("[$db_name][Chow-Liu] nodes/pars/models=$(num_nodes(pc_cl))/$(num_parameters(pc_cl))/$(log2(model_count(pc_cl)))")
    end
    write(myfile,"[$db_name][Chow-Liu] nodes/pars/models=$(num_nodes(pc_cl))/$(num_parameters(pc_cl))/$(log2(model_count(pc_cl)))\n")
    #@time
    pc_ff = fully_factorized_circuit(StructProbCircuit, vtree_cl);
    if VERBOSE
        println("[$db_name][Fully-Factorized] nodes/pars/models=$(num_nodes(pc_ff))/$(num_parameters(pc_ff))/$(log2(model_count(pc_ff)))")
    end
    write(myfile,"[$db_name][Fully-Factorized] nodes/pars/models=$(num_nodes(pc_ff))/$(num_parameters(pc_ff))/$(log2(model_count(pc_ff)))\n")
    #@time
    pc_st = learn_circuit(train_x; maxiter = STRUDEL_ITERATIONS, verbose = false)
    if VERBOSE
        println("[$db_name][Strudel] nodes/pars/models=$(num_nodes(pc_st))/$(num_parameters(pc_st))/$(log2(model_count(pc_st)))")
    end
    write(myfile,"[$db_name][Strudel] nodes/pars/models=$(num_nodes(pc_st))/$(num_parameters(pc_st))/$(log2(model_count(pc_st)))\n")
    #@time
    pc_sl = learn_circuit_slopp(vtree_cl,train_x,db_name,nc,thresh)
    #@time
    pc_sl = load_prob_circuit("$db_name-$nc-$thresh.psdd")
    @assert isstruct_decomposable(pc_sl) & isstruct_decomposable(pc_cl) & isstruct_decomposable(pc_st) & isstruct_decomposable(pc_ff)
    if VERBOSE
        println("[$db_name][SLoPP] nodes/pars/models=$(num_nodes(pc_sl))/$(num_parameters(pc_sl))/$(round(log2(model_count(pc_sl))))")
    end
    write(myfile,"[$db_name][SLoPP] nodes/pars/models=$(num_nodes(pc_sl))/$(num_parameters(pc_sl))/$(round(log2(model_count(pc_sl))))\n")
    #vtree_bal = Vtree(num_features(train_x), :balanced) # ex vtree2
    train_x2 = unique(train_x)
    tot = [0.0, 0.0, 0.0, 0.0]
    imp = [0.0, 0.0, 0.0]
    exc = 0
    indb_possible = 0
    indb_impossible = 0
    if !FAST
        @showprogress for j = 1:test_size
            if log_likelihood_avg(pc_sl, test_x[j:j,:]) != -Inf
                tot[1] += log_likelihood_avg(pc_sl, test_x[j:j,:])
                tot[2] += log_likelihood_avg(pc_ff, test_x[j:j,:])
                tot[3] += log_likelihood_avg(pc_cl, test_x[j:j,:])
                tot[4] += log_likelihood_avg(pc_st, test_x[j:j,:])
                if size(findall(i->i==true,[train_x2[k:k,:]==test_x[j:j,:] for k in 1:size(train_x2,1)]),1)>0
                    indb_possible += 1
                end
            else
                imp[1] += log_likelihood_avg(pc_ff, test_x[j:j,:])
                imp[2] += log_likelihood_avg(pc_cl, test_x[j:j,:])
                imp[3] += log_likelihood_avg(pc_st, test_x[j:j,:])
                exc += 1
                if size(findall(i->i==true,[train_x2[k:k,:]==test_x[j:j,:] for k in 1:size(train_x2,1)]),1)>0
                    indb_impossible += 1
                end
            end
        end
    end
    tot = tot/(test_size-exc)
    imp = imp/(exc)
    if VERBOSE
        println("[$db_name][average LL]SLOPP/FF/CL/ST=$tot")
        println("[$db_name][impossible LL]FF/CL/ST=$imp")
        println("[$db_name]exceptions=$exc,possible_indb=$indb_possible,impossible_indb=$indb_impossible")
    end
    write(myfile,"[$db_name][average LL]SLOPP/FF/CL/ST=$tot\n")
    write(myfile,"[$db_name][impossible LL]FF/CL/ST=$imp\n")
    write(myfile,"[$db_name]exceptions=$exc,possible_indb=$indb_possible,impossible_indb=$indb_impossible\n")
    close(myfile)
end
end

function write_psdd_file(s,v,filename)
    n = length(s)
    qq = Vector()  # DEBUG: Typing
    for decision_node = 1:n
        if s[decision_node].label == "D"
            vv = Int64[]
            for child = 1:n
                if s[child].parent == decision_node
                    push!(vv,child)
                end
            end
            push!(qq,[decision_node,vv])
        end
    end
    dictv = get_vtree2id(v)
    file_psdd = String[]
    for node = 1:n
        if s[node].label == "D"
            children = Int64[]
            children2 = Vector() # DEBUG: Typing
            children3 = Vector() # DEBUG: Typing
            for child = 1:n
                if s[child].parent == node
                    push!(children,child)
                end
            end
            n_paired_boxes = Int(size(children,1)/2)
            for qqq = 1:n_paired_boxes
                push!(children2,children[(qqq-1)*2+1]+1)
                push!(children2,children[(qqq-1)*2+2]+1)
                push!(children2,log(s[children[(qqq-1)*2+1]].theta))
                push!(children3,n-children[(qqq-1)*2+1]+1)
                push!(children3,n-children[(qqq-1)*2+2]+1)
                push!(children3,log(s[children[(qqq-1)*2+1]].theta))
            end
            push!(file_psdd,string(s[node].label," ",n-node+1," ",dictv[s[node].vtree]," ",n_paired_boxes," ",join(string.(children3)," ")),"\n")
        elseif s[node].label == "L"
            push!(file_psdd,string(s[node].label," ",n-node+1," ",dictv[s[node].vtree]," ",s[node].literal),"\n")
        else # Log
            push!(file_psdd,string(s[node].label," ",n-node+1," ",dictv[s[node].vtree]," ",s[node].literal," ",log(s[node].theta)),"\n")
        end
    end
    open("$filename.psdd", "w") do f
    write(f,"c\n")
    write(f,"psdd $n")
    write(f,join(reverse(file_psdd)))
    end
end

# Modified version of the Juice clustering to also retrieve the assignments of the clusters
function clustering4(data, mix_num::Int64; maxiter=200)::Vector{Int64}
    # n = num_examples(data)
    if mix_num == 1
        return ones(Int64,size(data,1))
    end
    data = Matrix(data)
    R = kmeans(data', mix_num; maxiter=maxiter) # DEBUG: initialization + maxiter
    @assert nclusters(R) == mix_num
    return assignments(R)
end

# Function to fix the col names with sub-datasets
function find_names(names_short::Vector{String},indexes_short::Vector{Int64},col_names::Vector{String})::Vector{Int64}
    m=Int64[]
    for i in indexes_short
        push!(m,findfirst(x->x==col_names[i], names_short))
    end
    return m
end

function learn_circuit_slopp(vtree::PlainVtree,dbase::DataFrame,dbname::String,n_cl::Int64,threshold::Int64)::Bool

    sdd = mynode[]
    titles = names(dbase)
    v = vtree
    d = dbase

    # weight=ones(Float64, num_examples(d))
    # (_, mi) = mutual_information(d, weight; α=0.0)
    # # vars = Var.(collect(1:num_features(d)))
    # info = to_long_mi(mi, MIN_INT, MAX_INT)
    # g = convert(SparseMatrixCSC, info)
    # partition = my_partition(my_graph(g), 2)
    # left_variables = findall(==(1), partition)
    # right_variables = findall(==(2), partition)
    root_node = mynode(vtree=v, db=d) #, left=left_variables, right=right_variables)

    push!(sdd,root_node)
    active = 1 # index of the decision node to be processed

    while active != nothing # stopping when no nodes to process are available
        v = sdd[active].vtree # current vtree
        d = sdd[active].db    # current db
        sdd[active].visited = true
        elements = expand_psdd(sdd[active],active,titles,n_cl,threshold)
        for e in elements
            push!(sdd,e)
        end
    active = findfirst(isequal(false),mappedarray((x) -> x.visited, sdd))
    # VTREE LEARNING
    # if LEARNVTREE
    # if !isnothing(active)
    # if length(sdd[active].left) == 0
    #     n_cols = size(sdd[active].db,2)
    #     mi2 = zeros(Float64, n_cols, n_cols)
    #     for k = 1:length(sdd)
    #         if names(sdd[k].db)==names(sdd[active].db)
    #             weight=ones(Float64, num_examples(sdd[k].db))
    #             (_, mi) = mutual_information(sdd[k].db, weight; α=0.0)
    #             mi2 = mi2 + mi
    #         end
    #     end
    #     # info2 = to_long_mi(mi2, MIN_INT, MAX_INT)
    #     # g2 = convert(SparseMatrixCSC, info2)
    #     # partition2 = my_partition(my_graph(g2), 2)
    #     # left_variables2 = findall(==(1), partition2)
    #     # right_variables2 = findall(==(2), partition2)
    #     # nm = names(sdd[active].db)
    #     # lnm = nm[left_variables2]
    #     # rnm = nm[right_variables2]
    #     # left_vars_partitioned = [parse(Int,q[2:end]) for q in lnm]
    #     # right_vars_partitioned = [parse(Int,q[2:end]) for q in rnm]
    #     for k = 1:length(sdd)
    #         if names(sdd[k].db)==names(sdd[active].db)
    #             sdd[k].left = left_vars_partitioned
    #             sdd[k].right = right_vars_partitioned
    #         end
    #     end
    # end
    # end
    # end
end

#for node in sdd
#    println(names(node.db),node.left,node.right)
#end
write_psdd_file(sdd,vtree,"$dbname-$n_cl-$threshold")
return true
end

#DATABASES_NAMES = ["nltcs","msnbc","kdd","plants","jester","bnetflix","baudio","accidents","tretail","pumsb_star",
#"dna","kosarek","msweb","tmovie","book","cwebkb","cr52","c20ng","ad","bbc","binarized_mnist"]
#for i = [1]#,3]
#    #@spawn
#    experiment(DATABASES_NAMES[i],1,500,0) # Best
#end
#@spawn
#experiment("nltcs",2,5000,0) # Best
#@spawn#experiment("msnbc",2,5000,0)
#@spawn experiment("kdd",2,5000,0)
#@spawn experiment("plants",2,5000,0)
#@spawn experiment("jester",2,5000,0)
#experiment("tretail",2,5000,0) #"Fast and good"
#@time experiment("pumsb_star",2,5000,0) #"Fast and good"
#@time experiment("kosarek",2,5000,0) # Fast and so and so provare 10000?
#@time experiment("msweb",2,5000,0) # Fast-medium and good
#@time  experiment("dna",2,5000,0) # Too good because of the few possible
#experiment("accidents",2,5000,0) #terzo class
#@time #experiment("bnetflix",1,5000,0) good
#experiment("baudio",2,5000,0) #bad as dna
#@time experiment("tmovie",1,2000,200) # Slow
#@time experiment("book",2,2000,200) # Slow
#@time experiment("cwebkb",2,2000,200) # Slow
#@timeexperiment("cr52",2,5000,0) # Slow-med Bad
#@time experiment("c20ng",2,2000,0) # Slow
#@time experiment("ad",2,5000,0) # Slow
#@time experiment("bbc",2,5000,0) #
#@time experiment("binarized_mnist",2,5000,0) #
#experiment("plants",2,5000,0) # Best
# Threads.@threads for db in db_names
#     train_x, valid_x, test_x = twenty_datasets(db)
#     println(" ",db,size(train_x),size(test_x))
# end
#c = kmeans(train_x, 2)#; rng = Random.seed!(2020))
#@time experiment("msnbc",1,1000) # Best



end # module
