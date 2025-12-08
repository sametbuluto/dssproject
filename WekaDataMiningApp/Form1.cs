using System;
using System.Windows.Forms;
using System.IO;
using System.Text;

namespace WekaDataMiningApp
{
    public partial class Form1 : Form
    {
        // UI Components declared here for simplicity in this file for now
        private Button btnLoad;
        private Button btnTrain;
        private Button btnDiscover;

        private TextBox txtResults;
        private Label lblStatus;
        
        // Discovery Inputs
        private TextBox txtSepalLen;
        private TextBox txtSepalWid;
        private TextBox txtPetalLen;
        private TextBox txtPetalWid;

        private WekaEngine _engine;

        public Form1()
        {
            InitializeComponent();
            SetupCustomUI();
            _engine = new WekaEngine();
        }

        private void SetupCustomUI()
        {
            this.Text = "Weka Data Mining App - Assignment 1";
            this.Size = new System.Drawing.Size(800, 600);
            this.StartPosition = FormStartPosition.CenterScreen;

            // Load Button
            btnLoad = new Button() { Text = "Load Dataset...", Location = new System.Drawing.Point(20, 20), Width = 150 };
            btnLoad.Click += BtnLoad_Click;
            this.Controls.Add(btnLoad);

            // Train Button
            btnTrain = new Button() { Text = "Run Tournament", Location = new System.Drawing.Point(340, 20), Width = 150 };
            btnTrain.Click += BtnTrain_Click;
            this.Controls.Add(btnTrain);

            // Results Box
            txtResults = new TextBox() { Location = new System.Drawing.Point(20, 60), Width = 740, Height = 300, Multiline = true, ScrollBars = ScrollBars.Vertical, ReadOnly = true, Font = new System.Drawing.Font("Consolas", 10F) };
            this.Controls.Add(txtResults);

            // Discovery Section
            var grpDiscover = new GroupBox() { Text = "Discover / Prediction (New Instance)", Location = new System.Drawing.Point(20, 380), Size = new System.Drawing.Size(740, 150) };
            this.Controls.Add(grpDiscover);

            int x = 20, y = 30;
            
            // Labels and TextBoxes
            grpDiscover.Controls.Add(new Label() { Text = "Sepal Length:", Location = new System.Drawing.Point(x, y), AutoSize = true });
            txtSepalLen = new TextBox() { Location = new System.Drawing.Point(x + 100, y - 3), Text = "5.1", Width = 60 };
            grpDiscover.Controls.Add(txtSepalLen);

            grpDiscover.Controls.Add(new Label() { Text = "Sepal Width:", Location = new System.Drawing.Point(x + 220, y), AutoSize = true  });
            txtSepalWid = new TextBox() { Location = new System.Drawing.Point(x + 320, y - 3), Text = "3.5", Width = 60 };
            grpDiscover.Controls.Add(txtSepalWid);

            y += 40;
            grpDiscover.Controls.Add(new Label() { Text = "Petal Length:", Location = new System.Drawing.Point(x, y), AutoSize = true  });
            txtPetalLen = new TextBox() { Location = new System.Drawing.Point(x + 100, y - 3), Text = "1.4", Width = 60 };
            grpDiscover.Controls.Add(txtPetalLen);

            grpDiscover.Controls.Add(new Label() { Text = "Petal Width:", Location = new System.Drawing.Point(x + 220, y), AutoSize = true  });
            txtPetalWid = new TextBox() { Location = new System.Drawing.Point(x + 320, y - 3), Text = "0.2", Width = 60 };
            grpDiscover.Controls.Add(txtPetalWid);

            // Discover Button
            btnDiscover = new Button() { Text = "Discover Class", Location = new System.Drawing.Point(x + 450, y - 10), Width = 150, Height = 40, BackColor = System.Drawing.Color.LightBlue };
            btnDiscover.Click += BtnDiscover_Click;
            grpDiscover.Controls.Add(btnDiscover);

            // Status Label
            lblStatus = new Label() { Text = "Ready. Please load a dataset.", Location = new System.Drawing.Point(20, 540), AutoSize = true };
            this.Controls.Add(lblStatus);
        }

        private void BtnLoad_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "ARFF Files (*.arff)|*.arff|All Files (*.*)|*.*";
            ofd.Title = "Select Dataset";
            
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                try 
                {
                    string path = ofd.FileName;
                    _engine.LoadData(path);
                    lblStatus.Text = $"Loaded: {Path.GetFileName(path)} | Instances: {_engine.GetInstanceCount()}";
                    txtResults.Text = $"Dataset Loaded successfully.\r\nFile: {path}\r\nRows: {_engine.GetInstanceCount()}\r\nClass Attribute: {_engine.GetClassAttributeName()}";
                }
                catch(Exception ex)
                {
                    MessageBox.Show("Error loading data: " + ex.Message);
                }
            }
        }

        private void BtnTrain_Click(object sender, EventArgs e)
        {
            if (_engine.GetInstanceCount() == 0)
            {
                MessageBox.Show("Please load a dataset first.");
                return;
            }

            lblStatus.Text = "Running Tournament (10 Algorithms)... Please wait.";
            txtResults.Text = "Running 10-Fold Cross Validation on 10 Algorithms...\r\n\r\n";
            Application.DoEvents(); // Force UI to render text before blocking

            try 
            {
                var results = _engine.RunTournament();
                
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("=== TOURNAMENT RESULTS ===");
                sb.AppendLine(String.Format("{0,-30} | {1,-15} | {2,-10}", "Algorithm", "Correctly Class.", "Accuracy"));
                sb.AppendLine(new string('-', 70));

                foreach (var r in results)
                {
                    sb.AppendLine(String.Format("{0,-30} | {1,-15} | {2,-10:F2}%", r.Name, r.CorrectlyClassified, r.Accuracy));
                }

                var best = _engine.GetBestResult();
                sb.AppendLine(new string('=', 70));
                sb.AppendLine($"WINNER: {best.Name}");
                sb.AppendLine($"Accuracy: {best.Accuracy:F2}% ({best.CorrectlyClassified} instances)");
                sb.AppendLine("Model saved for 'Discover' feature.");

                txtResults.Text = sb.ToString();
                
                // Show winner in status
                lblStatus.Text = $"Winner: {best.Name} ({best.Accuracy:F2}%)";
                MessageBox.Show($"Tournament Complete!\nBest Algorithm: {best.Name}\nCorrectly Classified: {best.CorrectlyClassified}", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                txtResults.Text += "\r\nError: " + ex.Message;
                MessageBox.Show("Error running tournament: " + ex.Message);
            }
        }

        private void BtnDiscover_Click(object sender, EventArgs e)
        {
            var best = _engine.GetBestResult();
            if (best == null)
            {
                 MessageBox.Show("Please run the algorithm comparison (Train) first!");
                 return;
            }

            try
            {
                double sl = double.Parse(txtSepalLen.Text);
                double sw = double.Parse(txtSepalWid.Text);
                double pl = double.Parse(txtPetalLen.Text);
                double pw = double.Parse(txtPetalWid.Text);

                // Use the best model for prediction
                string predictedClass = _engine.PredictWithBest(sl, sw, pl, pw);
                
                string msg = $"Using Winner Algorithm: {best.Name}\n\n";
                msg += $"Input: [{sl}, {sw}, {pl}, {pw}]\n";
                msg += $"Predicted Class: {predictedClass}";

                MessageBox.Show(msg, "Discovery Result", MessageBoxButtons.OK, MessageBoxIcon.Information);
                lblStatus.Text = $"Last Prediction: {predictedClass} (by {best.Name})";
            }
            catch (Exception ex)
            {
                 MessageBox.Show("Invalid Input. Please enter numbers.\n" + ex.Message);
            }
        }
    }
}
