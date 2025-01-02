#!/usr/bin/env python


def to_latex(bits, nbit, cols=10):
    v = len(bits) // cols
    rows = v if len(bits) % cols == 0 else v + 1
    for i, bit in enumerate(bits):
        if i == 0:
            print("\\midrule")
            print(
                "\\multirowcell{" + str(rows) + "}{\\textbf{" + str(nbit) + "-bit}} ",
                end="",
            )
        print(f"& {bit:.2f} ", end="")
        if (i + 1) % cols == 0:
            print(r" \\")
        elif i == len(bits) - 1:
            padding = " &" * (cols - (i + 1) % cols)
            print(f"{padding} \\\\")


def main(ncols=10):
    head = f"""
\\begin{{table*}}
\\centering
\\caption[SensiMiLP/KurtMiLP Experiment Settings]{{
The SensiMiLP/KurtMiLP dense experiment bit budget settings.
}}
\\label{{tab:sensi-kurt-milp-settings}}
\\begin{{tabular}}{{r{'l' * ncols}}}
\\toprule
"""

    heading = r"\textbf{Group} & "
    fields = [f" \\textbf{{V{i}}}" for i in range(1, ncols + 1)]
    heading += "&".join(fields)
    heading += r" \\"

    tail = r"""
\bottomrule
\end{tabular}
\end{table*}
"""

    with open("dense.txt", "r") as fh:
        bits = [float(f) for f in fh.readlines()]
        bits8 = [f for f in bits if f >= 8.0]
        bits7 = [f for f in bits if f >= 7.0 and f < 8.0]
        bits6 = [f for f in bits if f >= 6.0 and f < 7.0]
        bits5 = [f for f in bits if f >= 5.0 and f < 6.0]
        bits4 = [f for f in bits if f >= 4.0 and f < 5.0]
        bits3 = [f for f in bits if f >= 3.0 and f < 4.0]
        bits2 = [f for f in bits if f >= 2.0 and f < 3.0]

        print(head)
        print(heading)
        to_latex(bits8, 8, ncols)
        to_latex(bits7, 7, ncols)
        to_latex(bits6, 6, ncols)
        to_latex(bits5, 5, ncols)
        to_latex(bits4, 4, ncols)
        to_latex(bits3, 3, ncols)
        to_latex(bits2, 2, ncols)
        print(tail)


if __name__ == "__main__":
    main(12)
