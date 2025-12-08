using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using weka.core;
using weka.classifiers;
using weka.classifiers.bayes;
using weka.classifiers.trees;
using weka.classifiers.lazy;
using weka.classifiers.functions;
using weka.classifiers.rules;
using weka.classifiers.meta;
using weka.filters;
using java.io;

namespace WekaDataMiningApp
{
    public class AlgorithmResult
    {
        public string Name { get; set; }
        public double Accuracy { get; set; }
        public double CorrectlyClassified { get; set; }
        public Classifier Model { get; set; }
        public override string ToString()
        {
            return $"{Name} | Correct: {CorrectlyClassified} | Acc: {Accuracy:F2}%";
        }
    }

    public class WekaEngine
    {
        private Instances _instances;
        private AlgorithmResult _bestResult; // Stores the winner

        public void LoadData(string filePath)
        {
            var loader = new weka.core.converters.ArffLoader();
            loader.setFile(new File(filePath));
            _instances = loader.getDataSet();
            _instances.setClassIndex(_instances.numAttributes() - 1);
        }

        // Helper to create a pipeline: Filter + Classifier
        private Classifier CreatePipeline(Classifier baseClassifier, weka.filters.Filter filter)
        {
            if (filter == null) return baseClassifier;
            
            FilteredClassifier fc = new FilteredClassifier();
            fc.setFilter(filter);
            fc.setClassifier(baseClassifier);
            return fc;
        }

        // Helper to create a pipeline: Filter1 + Filter2 + Classifier
        private Classifier CreatePipeline(Classifier baseClassifier, weka.filters.Filter f1, weka.filters.Filter f2)
        {
            weka.filters.MultiFilter multi = new weka.filters.MultiFilter();
            multi.setFilters(new weka.filters.Filter[] { f1, f2 });
            
            FilteredClassifier fc = new FilteredClassifier();
            fc.setFilter(multi);
            fc.setClassifier(baseClassifier);
            return fc;
        }

        public List<AlgorithmResult> RunTournament()
        {
            if (_instances == null) throw new Exception("No data loaded.");

            List<AlgorithmResult> results = new List<AlgorithmResult>();

            // --- Define Filters ---
            // Normalize: Scales all numeric values to the [0, 1] range.
            var normalize = new weka.filters.unsupervised.attribute.Normalize();
            
            // Discretize: Converts numeric attributes to nominal.
            var discretize = new weka.filters.unsupervised.attribute.Discretize();
            
            // NominalToBinary: Converts nominal attributes to numeric (dummy encoding).
            var nomToBin = new weka.filters.unsupervised.attribute.NominalToBinary();

            // --- Configure 10 Classification Approaches ---

            // 1. Naive Bayes (Requires Nominal Data -> Applied Discretize Filter)
            results.Add(Evaluate("NaiveBayes (Discretized)", CreatePipeline(new NaiveBayes(), discretize)));

            // 2. Logistic Regression (Requires Numeric Data -> Applied Normalize + NominalToBinary Filters)
            results.Add(Evaluate("Logistic (Norm+Nom2Bin)", CreatePipeline(new Logistic(), nomToBin, normalize)));

            // 3. IBk (k=1) (KNN Requires Numeric + Normalized Data)
            var ibk1 = new IBk(); ibk1.setKNN(1);
            results.Add(Evaluate("IBk (k=1, Norm+Nom2Bin)", CreatePipeline(ibk1, nomToBin, normalize)));

            // 4. IBk (k=3)
            var ibk3 = new IBk(); ibk3.setKNN(3);
            results.Add(Evaluate("IBk (k=3, Norm+Nom2Bin)", CreatePipeline(ibk3, nomToBin, normalize)));

             // 5. IBk (k=5)
            var ibk5 = new IBk(); ibk5.setKNN(5);
            results.Add(Evaluate("IBk (k=5, Norm+Nom2Bin)", CreatePipeline(ibk5, nomToBin, normalize)));

            // 6. J48 (Decision Tree - Handles Native Data Types)
            results.Add(Evaluate("J48 (Raw)", new J48()));

            // 7. RandomTree
            results.Add(Evaluate("RandomTree (Raw)", new RandomTree()));

            // 8. REPTree
            results.Add(Evaluate("REPTree (Raw)", new REPTree()));

            // 9. SMO (Support Vector Machine - Requires Numeric + Normalized Data)
            results.Add(Evaluate("SMO (SVM, Norm+Nom2Bin)", CreatePipeline(new SMO(), nomToBin, normalize)));

            // 10. MultilayerPerceptron (Neural Network - Requires Numeric + Normalized Data)
            results.Add(Evaluate("MultilayerPerceptron (Norm+Nom2Bin)", CreatePipeline(new MultilayerPerceptron(), nomToBin, normalize)));

            // Determine Winner
            _bestResult = results.OrderByDescending(r => r.CorrectlyClassified).First();

            return results;
        }

        private AlgorithmResult Evaluate(string name, Classifier model)
        {
            try
            {
                // We use cross-validation (10-fold)
                Evaluation eval = new Evaluation(_instances);
                // Note: model.buildClassifier(_instances) is called internally by crossValidate
                // But FilteredClassifier needs to be built? evaluateModel does it.
                // CrossValidateModel does training and testing folds. 
                // However, to keep 'model' usable for future prediction, we usually Build it on full data afterwards,
                // or rely on the Fact that we need a trained model for "Discover".
                
                // 1. Cross Validate to get metrics
                java.util.Random rand = new java.util.Random(1);
                eval.crossValidateModel(model, _instances, 10, rand);

                // 2. Build model on FULL data for the "Discover" feature (Best practice for final deployment)
                model.buildClassifier(_instances);

                return new AlgorithmResult
                {
                    Name = name,
                    Accuracy = eval.pctCorrect(),
                    CorrectlyClassified = eval.correct(),
                    Model = model
                };
            }
            catch (Exception ex)
            {
                return new AlgorithmResult { Name = name + " (Error)", Accuracy = 0, CorrectlyClassified = 0, Model = null };
            }
        }

        public string PredictWithBest(double sl, double sw, double pl, double pw)
        {
            if (_bestResult == null || _bestResult.Model == null) return "No model trained yet.";

            try
            {
                // Create attributes array matching the dataset
                double[] vals = new double[_instances.numAttributes()];
                
                // Assuming Iris strict structure: SepalL, SepalW, PetalL, PetalW, Class
                // We need to match inputs to Attribute Indices.
                // Verify order from your file content check: 
                // 0: sepallength, 1: sepalwidth, 2: petallength, 3: petalwidth, 4: class
                vals[0] = sl;
                vals[1] = sw;
                vals[2] = pl;
                vals[3] = pw;
                vals[4] = Instance.missingValue();

                // Legacy Instance
                Instance inst = new Instance(1.0, vals);
                inst.setDataset(_instances);

                double resultClassIndex = _bestResult.Model.classifyInstance(inst);
                return _instances.classAttribute().value((int)resultClassIndex);
            }
            catch (Exception ex)
            {
                return "Prediction Error: " + ex.Message;
            }
        }

        public int GetInstanceCount() => _instances == null ? 0 : _instances.numInstances();
        public string GetClassAttributeName() => _instances == null ? "?" : _instances.classAttribute().name();
        
        public AlgorithmResult GetBestResult() => _bestResult;
    }
}
