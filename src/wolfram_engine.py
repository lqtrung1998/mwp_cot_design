import re
from wolframclient.language import wl, wlexpr
from wolframclient.evaluation import WolframLanguageSession
import os
from src.utils import timeout

def init_session(session):
    script = [
              'Clear["Global`*"]',
              "Slope[expr_] := (grad = Grad[expr[[2]] - expr[[1]], {x,y}]; -grad[[1]]/grad[[2]])",
              "YIntercept[expr_] := (y /. Solve[expr /. {x -> 0}, y])[[1]]",
              "SlopeLineForm[expr_] := y == Slope[expr] * x + YIntercept[expr]",
              "GraphPointDistance[expr_, p_] := RegionDistance[ImplicitRegion[expr, {x,y}], p]",
              "StandardLineForm[expr_] := -Slope[expr] * x + y - YIntercept[expr] == 0",
              "SelectOne[expr_] := If[Head @ expr === List && Length[expr] == 1, expr[[1]], expr]",
              'CompleteSquare[expr_, var_] := ResourceFunction["CompleteSquare"][expr, x]',
              "AllVariables[x_] := (output = If[Head @ x === Symbol, {x}, DeleteDuplicates@Cases[x, _Symbol, -1]]; output = DeleteCases[output, True]; output = DeleteCases[output, False])",
              "CheckIneq[x_] := Length[Cases[{x}, _Less|_Greater|_LessEqual|_GreaterEqual|_Inequality|_Unequal, Infinity]] > 0",
    ]
    session.evaluate(wlexpr(";".join(script)))

def solve(session, query, replace_underscore=True):
    if replace_underscore:
        query = query.replace("_", "")
        
    init_session(session)
    scripts = []
    for line in query.split("\n"):
        code = re.sub("\(\*.+\*\)", "", line.strip()).strip()
        if not code:
            continue
        if replace_underscore:
            if "/." in code or "Keys" in code:
                var = re.search("^[A-Za-z]+ \= ", code)
                if var:
                    var = var.group(0)
                    code = code.replace(var, "%sSelectOne[" % var) + "]"
                else:
                    code = "SelectOne[%s]" % code
        else:
            if "/." in code or "Keys" in code:
                var = re.search("^v[0-9]+ \= ", code)
                if var:
                    var = var.group(0)
                    code = code.replace(var, "%sSelectOne[" % var) + "]"
                else:
                    code = "SelectOne[%s]" % code
        scripts.append(code)

    result = session.evaluate(wlexpr("ToString[SelectOne[TimeConstrained[(%s), 10]], InputForm]" % ";".join(scripts)))
    script = "tmpres = %s; If[Head @ tmpres === Quantity, tmpres[[1]], tmpres]" % result
    result = session.evaluate(wlexpr("TimeConstrained[ToString[%s, InputForm], 10]" % script))
    return result

session = None

def run_wolfram_code(code_gen):
    # WolframKernel can only run with single process, if the code hangs, previous session need to be cleared.
    global session
    if session is None:
        os.system("kill -9 $(ps aux | grep WolframKernel | awk '{print $2}')")
        session = WolframLanguageSession()
    try:
        with timeout(seconds=10):
            result = solve(session, code_gen)
    except Exception as e:
        os.system("kill -9 $(ps aux | grep WolframKernel | awk '{print $2}')")
        session = WolframLanguageSession()
        raise
    return result
