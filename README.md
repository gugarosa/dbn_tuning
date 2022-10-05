# Fine-Tuning Deep Belief Networks with Harmony-Based Optimization

*This repository holds all the necessary code to run the very-same experiments described in the chapter "Fine-Tuning Deep Belief Networks with Harmony-Based Optimization".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure
 * `core`
   * `dataset.py`: Customized dataset class without logging;
   * `dbn.py`: Customized DBN class adapted to optimization pipeline;
 * `utils`
   * `loader.py`: Utility to load datasets and split them into training, validation and testing sets;
   * `objects.py`: Wraps objects instantiation for command line usage;
   * `optimizer.py`: Wraps the optimization task into a single method;  
   * `target.py`: Implements the objective functions to be optimized.
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

In order to run the experiments, you can use `torchvision` to load pre-implemented datasets.

---

## Usage

### Model Optimization

The first step is to conduct the optimization pipeline over the DBN architecture. Basically, agents encode the desired hyperparameters and attempt to find the minimum reconstruction error over the validation set. At the end of the optimization, such values are stored into a file for further usage. To accomplish such a step, one needs to use the following script:

```Python
python dbn_optimization.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate arguments.*

### Model Evaluation

After conducting the optimization task, one needs to gather the best hyperparameters from file and re-train a DBN. Please, use the following script to accomplish such a procedure:

```Python
python dbn_evaluation.py -h
```

*Note that the evaluation step is the only script that uses the testing data.*

### Inspect Optimization History (Optional)

Additionally, one can gather the optimization history files and input them to a script that analyzes its convergence and produces a plot that compares how each optimization technique has performed during its procedure. Please, use such a script as follows:

```Python
python inspect_history.py -h
```

*Note that one needs to input the history files without their seed and extension, e.g., `HS_1ag_12var_2it`.*

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this chapter. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository.

---
