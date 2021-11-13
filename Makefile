SAGE=sage

install:
	$(SAGE) -pip install --upgrade --no-index --use-feature=in-tree-build -v .

install-user:
	$(SAGE) -pip install --upgrade --no-index --use-feature=in-tree-build -v --user .

uninstall:
	$(SAGE) -pip uninstall momentproblems

test:
	$(SAGE) -t -l -p --random-seed=`$(SAGE) -c "print(randint(0, 10**6))"` momentproblems

doc:
	cd docs && $(SAGE) -sh -c "make html"

doc-clean:
	cd docs && rm -rf _build/

example1:
	$(SAGE) -python -c "from momentproblems.examples import example1;\
	    from sage.all import set_random_seed; set_random_seed(502350);\
	    Gs = example1.generate_plots(r=3, D=4);\
	    from sage.all import set_random_seed; set_random_seed(27286);\
	    Gs += example1.generate_plots();\
	    [Gk.save(f'example1_{k}.pdf') for k,Gk in enumerate(Gs)]"
example2:
	$(SAGE) -python -c "from momentproblems.examples import example2;\
	    Gs = example2.generate_plots(r=6);\
	    [Gk.save(f'example2_{k}.pdf') for k,Gk in enumerate(Gs)]"
# seed for reproducible print-friendly example
example2b:
	$(SAGE) -python -c "from momentproblems.examples import example2;\
	    from sage.all import set_random_seed; set_random_seed(11);\
	    Gs = example2.generate_plots2(r=3);\
	    [Gk.save(f'example2b_{k}.pdf') for k,Gk in enumerate(Gs)]"
example3:
	$(SAGE) -python -c "from momentproblems.examples import example3;\
	    Gs = example3.generate_plots();\
	    [Gk.save(f'example3_{k}.pdf') for k,Gk in enumerate(Gs)]"
example3b:
	$(SAGE) -python -c "from momentproblems.examples import example3;\
	    example3.generate_plots(filename='example3.dat');"
