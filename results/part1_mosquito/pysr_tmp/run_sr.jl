
using SymbolicRegression, DelimitedFiles

X_raw = readdlm("/root/wenmei/results/part1_mosquito/pysr_tmp/X.csv", ',', Float64)   # [rows, 6]
y     = vec(readdlm("/root/wenmei/results/part1_mosquito/pysr_tmp/y.csv", ',', Float64))
X     = permutedims(X_raw)                       # [6, rows]

options = SymbolicRegression.Options(
    binary_operators=[+, -, *, /],
    unary_operators=[exp, cos, safe_sqrt],
    maxsize=25,
    populations=30,
    timeout_in_seconds=300,
)

hall = equation_search(X, y;
    options=options,
    niterations=300,
    variable_names=["T", "H", "R", "Tm", "Hm", "Rm"],
)

# Save hall of fame
dominating = calculate_pareto_frontier(hall)
open("/root/wenmei/results/part1_mosquito/pysr_tmp/hall_of_fame.csv", "w") do io
    println(io, "complexity,loss,equation")
    for member in dominating
        c = compute_complexity(member, options)
        l = member.loss
        eq = string_tree(member.tree, options)
        println(io, "$c,$l,\"$eq\"")
    end
end

# Save best (last) equation text
best = dominating[end]
best_eq = string_tree(best.tree, options)
open("/root/wenmei/results/part1_mosquito/pysr_tmp/best_formula.txt", "w") do io
    println(io, best_eq)
end

# Evaluate ALL Pareto members on X for later selection
open("/root/wenmei/results/part1_mosquito/pysr_tmp/pareto_preds.csv", "w") do io
    for (idx, member) in enumerate(dominating)
        preds, ok = eval_tree_array(member.tree, X, options)
        if !ok
            preds = fill(NaN, size(X, 2))
        end
        println(io, join(preds, ","))
    end
end

println("DONE: best = ", best_eq)
