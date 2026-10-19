#!/bin/sh
# Rebuilds ../Onieal_Assignment5.docx from Onieal_Assignment5.md. Run from writeup/ with pandoc and graphviz installed.
# The assignment's own figures come out of ../Assignment5.docx. reference.docx holds the Word styles,
# including "Answer Text", the blue style the answers use. export_pdf.ps1 then makes the PDF with Word.
set -e
pandoc ../Assignment5.docx -t markdown --extract-media=. -o /dev/null
neato -n2 -Tpng -Gdpi=220 figures/rnn_graph.dot -o figures/rnn_graph.png
pandoc Onieal_Assignment5.md -o ../Onieal_Assignment5.docx --reference-doc=reference.docx --resource-path=.:..
