module LearnPSDDasSPN

using Printf
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
using Random
import Base.Threads.@spawn

Random.seed!(1234); # Clustering Reproducibility (kmeans needs a random init)
STRUDEL_ITERATIONS1 = 50
STRUDEL_ITERATIONS2 = 500
DEBUGGING_MODE = true
LEARN_VTREE = false
VERBOSE = true
δINT = 999999
MIN_INT = 1
MAX_INT = δINT + MIN_INT

@with_kw mutable struct mynode
    vtree::PlainVtree # DEBUG: remove, use vtree index only
    db::DataFrame = DataFrame() # DEBUG: replace by row indexes only
    label::String = "D" # Label
    position::String = "" # DEBUG: prime or sub, to remove in the final version
    parent::Int64 = -1
    literal::Int64 = 0
    visited::Bool = false
    theta::Float64 = -1.0
    left::Vector{Int64} = []
    right::Vector{Int64} = []
    vindex::Int64 = 0
end

struct WeightedGraph # From Juice DEBUG: replace with inclusion
    nvtxs::idx_t
    xadj::Vector{idx_t}
    adjncy::Vector{idx_t}
    adjwgt::Vector{idx_t} # edge weights
    WeightedGraph(nvtxs, xadj, adjncy, adjwgt) = new(nvtxs, xadj, adjncy, adjwgt)
end

function my_graph(G::SparseMatrixCSC; check_hermitian=true)  # From Juice DEBUG: replace with inclusion
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

function my_partition(G::WeightedGraph, nparts::Integer)  # From Juice DEBUG: replace with inclusion
    part = Vector{Metis.idx_t}(undef, G.nvtxs)
    edgecut = fill(idx_t(0), 1)
    Metis.METIS_PartGraphRecursive(G.nvtxs, idx_t(1), G.xadj, G.adjncy, C_NULL, C_NULL, G.adjwgt,
                                 idx_t(nparts), C_NULL, C_NULL, Metis.options, edgecut, part)
    return part
end

function clustering_can_be_done(n_rows::Int64,n_cols::Int64, n_clsts::Int64,threshold_rows::Int64,threshold_cols::Int64)
    return n_rows >= threshold_rows && n_cols <= threshold_cols && n_clsts > 1
end

function expand_psdd(nd::mynode,active::Int64,titles::Vector{String},ncl::Int64,thr1::Int64,thr2::Int64)
    #if !LEARN_VTREE # DEBUG: other way round
    l = typeof(nd.vtree.left) == PlainVtreeLeafNode ? [Int(variable(nd.vtree.left))] : [Int(n) for n in variables(nd.vtree.left)]
    r = typeof(nd.vtree.right) == PlainVtreeLeafNode ? [Int(variable(nd.vtree.right))] : [Int(n) for n in variables(nd.vtree.right)]
    #else
    #    l = nd.left
    #    r = nd.right
    #end
    dl = nd.db[:,find_names(names(nd.db),l,titles)]
    dr = nd.db[:,find_names(names(nd.db),r,titles)]
    #if length(nd.left) > 1 # Many vars on the left
    if typeof(nd.vtree.left) == PlainVtreeInnerNode # Many vars on the left
        if clustering_can_be_done(nrow(unique(dl)),length(l),ncl,thr1,thr2)
            return expand_many_left_multiclusters(nd::mynode,active::Int64,dl,dr,ncl) #return n_c left and right clusters
        else
            return expand_many_left_singlecluster(nd::mynode,active::Int64,dl,dr)
        end
    else
            #if length(nd.right) > 1 # Many variables on the right
            if typeof(nd.vtree.right) == PlainVtreeInnerNode # Many variables on the right
            return expand_single_left_many_right(nd::mynode,active::Int64,dl,dr)
        else
            return expand_single_left_one_right(nd::mynode,active::Int64,dl,dr)
        end
    end
end

function expand_many_left_multiclusters(nd::mynode,active::Int64,d_left::DataFrame,d_right::DataFrame,n_cl::Int64)
    elements = []
    indx_l = clustering4(d_left, n_cl) # DEBUG: import from Juice
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

function experiment(db_name::String , nc::Int64, rows_threshold::Int64 , cols_threshold::Int64 , test_size::Int64=-1, topdown::Bool=false)

    train_x, valid_x, test_x = twenty_datasets(db_name)
    test_size > 0 && (test_x = test_x[1:test_size,:]) # Smaller test sets for quick debugging

    row_db = @sprintf("%s,n_f=%i,n_train=%i(%i),n_test=%i(%i),top-down=%i,",db_name, size(test_x,2),size(train_x,1),size(unique(train_x),1),size(test_x,1),size(unique(test_x),1),topdown)
    row_pars = @sprintf("n_c=%i,threshold_row=%i,threshold_col=%i\n",nc,rows_threshold,cols_threshold)

    VERBOSE && print(row_db,row_pars)

    if topdown
        t_vtree = @elapsed vtree = learn_vtree(train_x; alg=:topdown)
    else
        t_vtree = @elapsed pc_cl, vtree = learn_chow_liu_tree_circuit(train_x)
    end
    if true #!DEBUGGING_MODE
        t_st1 = @elapsed pc_st1 = learn_circuit(train_x; maxiter = STRUDEL_ITERATIONS1, verbose = false)
        t_st2 = @elapsed pc_st2 = learn_circuit(train_x; maxiter = STRUDEL_ITERATIONS2, verbose = false)
        row_str1 = @sprintf("%s,strudel%i_nodes/pars=%i/%i,t1=%g,",db_name,STRUDEL_ITERATIONS1,num_nodes(pc_st1),num_parameters(pc_st1),t_st1)
        row_str2 = @sprintf("strudel%i_nodes/pars=%i/%i,t2=%g\n",STRUDEL_ITERATIONS2,num_nodes(pc_st2),num_parameters(pc_st2),t_st2)
        VERBOSE && print(row_str1,row_str2)

        t_sl = @elapsed learn_circuit_slopp(vtree,train_x,db_name,nc,rows_threshold,cols_threshold,topdown)
        t_sl2 = @elapsed pc_sl = load_prob_circuit("$db_name-$nc-$rows_threshold-$cols_threshold-$topdown.psdd")
        t_slopp = (t_vtree+t_sl+t_sl2)
        g_st1 = num_parameters(pc_st1)
        g_st2 = num_parameters(pc_st2)
        g_slopp = num_parameters(pc_sl)
        @assert isstruct_decomposable(pc_sl)
        row_sl = @sprintf("%s,SLoPP_nodes/pars/mc=%i/%i/%g,t=%g,%i\n",db_name,num_nodes(pc_sl),num_parameters(pc_sl),round(log2(model_count(pc_sl)),digits=3),t_slopp,isdeterministic(pc_sl))
        VERBOSE && print(row_sl)
    end

    train_x2 = unique(train_x)
    ll_comp = zeros(3)
    ll_incomp = zeros(2)
    incomp_instances = 0
    indb_instances = 0
    if true #!DEBUGGING_MODE
         @showprogress for j = 1:size(test_x,1)
            compatible = (log_likelihood_avg(pc_sl, test_x[j:j,:]) != -Inf)
            compatible ? ll_comp[1] += log_likelihood_avg(pc_st1, test_x[j:j,:]) : ll_incomp[1] += log_likelihood_avg(pc_st1, test_x[j:j,:])
            compatible ? ll_comp[2] += log_likelihood_avg(pc_st2, test_x[j:j,:]) : ll_incomp[2] += log_likelihood_avg(pc_st2, test_x[j:j,:])
            compatible && (ll_comp[3] += log_likelihood_avg(pc_sl, test_x[j:j,:]))
            (!compatible) && (incomp_instances += 1)
            size(findall(i->i==true,[train_x2[k:k,:]==test_x[j:j,:] for k in 1:size(train_x2,1)]),1)>0 && (indb_instances += 1)
         end
     end

    if true #!DEBUGGING_MODE
        ll_comp /= (size(test_x,1)-incomp_instances)
        ll_incomp /= incomp_instances
        row_avg_ll_comp = @sprintf("%s,comp_ST1/2/SLoPP=%g,%g,%g,",db_name,ll_comp[1],ll_comp[2],ll_comp[3])
        row_avg_ll_incomp = @sprintf("incomp_ST1/2=%g,%g,incompatible=%i,indb=%d\n",ll_incomp[1],ll_incomp[2],incomp_instances,indb_instances)
        VERBOSE && println(row_avg_ll_comp,row_avg_ll_incomp)
        open("summary_$db_name-$nc-$rows_threshold-$cols_threshold-$test_size-$topdown.results", "w") do myfile
        write(myfile,row_db,row_pars,row_str1,row_str2,row_sl,row_avg_ll_comp,row_avg_ll_incomp)
        gamma = incomp_instances
        gamma = round(incomp_instances/size(test_x,1)*100.0;digits=2)
        s1 = @sprintf("summary,%s,%g,%g,%g,%g,",db_name,ll_comp[1],ll_comp[2],ll_comp[3],gamma) # round(log2(model_count(pc_sl))/size(test_x,2)*100.0;digits=2)
        s2 = @sprintf "%g,%g," round(t_st1/t_slopp;digits=2) round(t_st2/t_slopp;digits=2)
        s3 = @sprintf "%g,%g," round(g_st1/g_slopp;digits=2) round(g_st2/g_slopp;digits=2)
        s4 = @sprintf "%g,%g\n" round(ll_incomp[1];digits=2) round(ll_incomp[2];digits=2)
        println(s1,s2,s3,s4)
        write(myfile,s1,s2,s3,s4)
        close(myfile)
    end
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
function clustering4(data, mix_num::Int64)::Vector{Int64}
    # n = num_examples(data)
    if mix_num == 1
        return ones(Int64,size(data,1))
    end
    data = Matrix(data)
    R = kmeans(data', mix_num; maxiter=100) #, init=:kmcen)
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


function partitioner(db::DataFrame)
    weight=ones(Float64, num_examples(db))
    (_, mi) = mutual_information(db, weight; α=0.0)
    # # # vars = Var.(collect(1:num_features(d)))
    info = to_long_mi(mi, MIN_INT, MAX_INT)
    g = convert(SparseMatrixCSC, info)
    partition = my_partition(my_graph(g), 2)
    left_variables = findall(==(1), partition)
    right_variables = findall(==(2), partition)
return left_variables,right_variables
end

function learn_circuit_slopp(vtree::PlainVtree,dbase::DataFrame,dbname::String,n_cl::Int64,r_threshold::Int64,c_threshold::Int64, topdown::Bool)::Bool

    sdd = mynode[]
    titles = names(dbase)
    v = vtree
    d = dbase
    vvv = [] # experimental

    if false #LEARN_VTREE
        left_variables, right_variables = partitioner(d)
        root_node = mynode(vtree=v, db=d, left=left_variables, right=right_variables)
        push!(vvv,[left_variables,right_variables]) # experimental
    else
        root_node = mynode(vtree=v, db=d)
    end

    push!(sdd,root_node)
    active = 1 # index of the decision node to be processed

    while active != nothing # stopping when no nodes to process are available
        v = sdd[active].vtree # current vtree
        d = sdd[active].db    # current db
        sdd[active].visited = true
        elements = expand_psdd(sdd[active],active,titles,n_cl,r_threshold,c_threshold)
        for e in elements
            push!(sdd,e)
        end
        active = findfirst(isequal(false),mappedarray((x) -> x.visited, sdd))
    end
    # if false #LEARN_VTREE
    # if !isnothing(active)
    # if length(sdd[active].left) == 0
    #      n_cols = size(sdd[active].db,2)
    #      #if n_cols > 1
    #          #println(n_cols," xx x xxx x ")
    #      mi2 = zeros(Float64, n_cols, n_cols)
    #      for k = 1:length(sdd)
    #          if names(sdd[k].db)==names(sdd[active].db)
    #              weight=ones(Float64, num_examples(sdd[k].db))
    #              (_, mi) = mutual_information(sdd[k].db, weight; α=0.0)
    #              mi2 = mi2 + mi
    #          end
    #      end
    #      info2 = to_long_mi(mi2, MIN_INT, MAX_INT) # DEBUG: check max_int
    #      g2 = convert(SparseMatrixCSC, info2)
    #      partition2 = my_partition(my_graph(g2), 2)
    #      println(partition2)
    #      left_variables2 = findall(==(1), partition2)
    #      right_variables2 = findall(==(2), partition2)
    #      nm = names(sdd[active].db)
    #      lnm = nm[left_variables2]
    #      rnm = nm[right_variables2]
    #      left_vars_partitioned = [parse(Int,q[2:end]) for q in lnm]
    #      right_vars_partitioned = [parse(Int,q[2:end]) for q in rnm]
    #      push!(vvv,[left_vars_partitioned,right_vars_partitioned]) # experimental
    #      for k = 1:length(sdd)
    #          if names(sdd[k].db)==names(sdd[active].db)
    #              sdd[k].left = left_vars_partitioned
    #              sdd[k].right = right_vars_partitioned
    #          end
    #      end
    #  end
    # #end
    # end
    # end

    # "$db_name-$nc-$rows_threshold-$cols_threshold-$test_size-$topdown.results"
    file_id = "$dbname-$n_cl-$r_threshold-$c_threshold-$topdown"
    write_psdd_file(sdd,vtree,file_id)
    return true
end

DATABASES_NAMES = ["book"] #"kdd", "nltcs"]
#"nltcs",,"kdd","plants","jester","bnetflix","baudio","accidents","tretail","pumsb_star","dna","kosarek","msweb","tmovie","book","cwebkb","cr52","c20ng","ad","bbc"]


#@sync begin
for db_name in DATABASES_NAMES
#@spawn begin
            println(db_name)
            @time experiment(db_name,2,3,2,0,false)
            @time experiment(db_name,2,3,2,0,true)
#          end
end
#end
end
