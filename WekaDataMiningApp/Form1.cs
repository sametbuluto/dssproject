using System;
using System.Windows.Forms;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Drawing;

namespace WekaDataMiningApp
{
    public partial class Form1 : Form
    {
        private Button btnLoad;
        private Button btnTrain;
        private TextBox txtResults;
        private Label lblStatus;
        
        private GroupBox grpDiscover;
        private Panel pnlDynamicInputs;
        private Button btnDiscover;
        
        private WekaEngine _engine;
        private List<Control> _dynamicInputControls;
        private List<Label> _dynamicLabels;

        public Form1()
        {
            InitializeComponent();
            SetupCustomUI();
            _engine = new WekaEngine();
            _dynamicInputControls = new List<Control>();
            _dynamicLabels = new List<Label>();
        }

        private void SetupCustomUI()
        {
            this.Text = "Weka Data Mining App - Assignment 1";
            this.Size = new Size(800, 600);
            this.StartPosition = FormStartPosition.CenterScreen;

            // Load Button
            btnLoad = new Button() { Text = "Load Dataset...", Location = new Point(20, 20), Width = 150 };
            btnLoad.Click += BtnLoad_Click;
            this.Controls.Add(btnLoad);

            // Train Button
            btnTrain = new Button() { Text = "Run Tournament", Location = new Point(340, 20), Width = 150 };
            btnTrain.Click += BtnTrain_Click;
            this.Controls.Add(btnTrain);

            // Results Box
            txtResults = new TextBox() { Location = new Point(20, 60), Width = 740, Height = 300, Multiline = true, ScrollBars = ScrollBars.Vertical, ReadOnly = true, Font = new Font("Consolas", 10F) };
            this.Controls.Add(txtResults);

            // Discovery Section
            grpDiscover = new GroupBox() { Text = "Discover / Prediction (New Instance)", Location = new Point(20, 380), Size = new Size(740, 150) };
            this.Controls.Add(grpDiscover);

            // Dynamic inputs panel inside groupbox
            pnlDynamicInputs = new Panel() { Location = new Point(10, 25), Size = new Size(620, 90), AutoScroll = true };
            grpDiscover.Controls.Add(pnlDynamicInputs);

            // Discover Button
            btnDiscover = new Button() { Text = "Discover Class", Location = new Point(640, 50), Width = 90, Height = 40, BackColor = Color.LightBlue, Enabled = false };
            btnDiscover.Click += BtnDiscover_Click;
            grpDiscover.Controls.Add(btnDiscover);

            // Status Label
            lblStatus = new Label() { Text = "Ready. Please load a dataset.", Location = new Point(20, 540), AutoSize = true };
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
                    
                    // Generate dynamic inputs based on dataset
                    GenerateDynamicInputs();
                }
                catch(Exception ex)
                {
                    MessageBox.Show("Error loading data: " + ex.Message);
                }
            }
        }

        private void GenerateDynamicInputs()
        {
            // Clear previous controls
            pnlDynamicInputs.Controls.Clear();
            _dynamicInputControls.Clear();
            _dynamicLabels.Clear();

            var attributes = _engine.GetAttributeInfo();
            
            int x = 10, y = 10;
            int labelWidth = 100;
            int controlWidth = 120;
            int columnSpacing = 250;

            int column = 0;
            int row = 0;

            foreach (var attr in attributes)
            {
                int currentX = x + (column * columnSpacing);
                int currentY = y + (row * 35);

                // Label
                Label lbl = new Label() { Text = attr.Name + ":", Location = new Point(currentX, currentY + 3), Width = labelWidth, AutoSize = false };
                pnlDynamicInputs.Controls.Add(lbl);
                _dynamicLabels.Add(lbl);

                Control inputControl;

                if (attr.IsNominal)
                {
                    // ComboBox for nominal
                    ComboBox cmb = new ComboBox() { Location = new Point(currentX + labelWidth, currentY), Width = controlWidth, DropDownStyle = ComboBoxStyle.DropDownList };
                    cmb.Items.AddRange(attr.PossibleValues.ToArray());
                    if (cmb.Items.Count > 0) cmb.SelectedIndex = 0;
                    inputControl = cmb;
                }
                else
                {
                    // TextBox for numeric
                    TextBox txt = new TextBox() { Location = new Point(currentX + labelWidth, currentY), Width = controlWidth, Text = "0" };
                    inputControl = txt;
                }

                inputControl.Tag = attr;
                pnlDynamicInputs.Controls.Add(inputControl);
                _dynamicInputControls.Add(inputControl);

                // Arrange in 2 columns
                column++;
                if (column >= 2)
                {
                    column = 0;
                    row++;
                }
            }

            btnDiscover.Enabled = false; // Will be enabled after tournament
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
            Application.DoEvents();

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
                
                lblStatus.Text = $"Winner: {best.Name} ({best.Accuracy:F2}%)";
                btnDiscover.Enabled = true;
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
                // Collect values from dynamic controls
                object[] values = new object[_dynamicInputControls.Count];

                for (int i = 0; i < _dynamicInputControls.Count; i++)
                {
                    Control ctrl = _dynamicInputControls[i];
                    AttributeInfo attr = ctrl.Tag as AttributeInfo;

                    if (attr.IsNominal)
                    {
                        ComboBox cmb = ctrl as ComboBox;
                        values[i] = cmb.SelectedItem?.ToString() ?? "";
                    }
                    else
                    {
                        TextBox txt = ctrl as TextBox;
                        values[i] = txt.Text;
                    }
                }

                string predictedClass = _engine.PredictWithBest(values);
                
                string msg = $"Using Winner Algorithm: {best.Name}\n\n";
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
