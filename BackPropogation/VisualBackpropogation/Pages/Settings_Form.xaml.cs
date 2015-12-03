using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using VisualBackPropogation.Custom_Controls;

namespace VisualBackPropogation.Pages
{
    /// <summary>
    /// Interaction logic for Settings_Form.xaml
    /// </summary>
    public partial class Settings_Form : Page
    {
        private UIElement[] Elements;
        public Settings_Form()
        {
            InitializeComponent();
        }

        public void LoadFromFile(object sender, RoutedEventArgs e)
        {

            Load_File_Info(LoadFile());
            

        }

        //Load the file infromation from a known file
        public void Load_File_Info(string filename)
        {
            if (Elements == null)
            {
                createElementList();
            }
            if (filename != null)
            {
                string[] lines = System.IO.File.ReadAllLines(filename);
                string[] line;
                for (int k = 0, i = 0; k < lines.Length; k++)
                {
                    line = lines[k].Split();
                    if (!line[0].StartsWith("Type"))
                    {
                        if (Elements[i] is File_Select_Button)
                        {
                            ((File_Select_Button)Elements[i]).Text = line[1];
                        }
                        else if (Elements[i] is Number_Picker)
                        {
                            ((Number_Picker)Elements[i]).Text = line[1];
                        }
                        else if (Elements[i] is True_False_Button)
                        {
                            ((True_False_Button)Elements[i]).Text = line[1];
                        }
                        else if (Elements[i] is TextBox)
                        {
                            ((TextBox)Elements[i]).Text = line[1];

                        }
                        i++;
                    }

                }
            }
        }

        public string LoadFile()
        {
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();

            //Filter the file type
            dlg.DefaultExt = ".txt";
            dlg.Filter = "TXT Files (*.txt)| *.txt";
            // Display OpenFileDialog by calling ShowDialog method 
            Nullable<bool> result = dlg.ShowDialog();
            if (result==true)
            {
                return dlg.FileName;
            }
            else
            {
                return null;
            }

        }

        private void CreateSettingsFile(object sender, RoutedEventArgs e)
        {

            // Create OpenFileDialog 
            Microsoft.Win32.SaveFileDialog dlg = new Microsoft.Win32.SaveFileDialog();



            // Set filter for file extension and default file extension 
            dlg.DefaultExt = ".txt";
            dlg.Filter = "TXT Files (*.txt)|*.txt";


            // Display OpenFileDialog by calling ShowDialog method 
            Nullable<bool> result = dlg.ShowDialog();



            // Get the selected file name and display in a TextBox 
            if (result == true)
            {
                // Open document 
                string filename = dlg.FileName;


                //List of Values
                string[] lines = {"s_network_name",
                                 "i_loops",
                                 "i_number_allowed_failures",
                                 "i_number_before_growth_potential",
                                 "i_number_allowed_same","i_input",
                                 "i_output","d_threshold",
                                 "d_distance_threshold",
                                 "d_neuron_distance_threshold",
                                 "d_row_success_threshold",
                                 "d_neuron_success_threshold",
                                 "d_fluctuate_square_mean",
                                 "b_allow_node_locking",
                                 "d_lock_node_level",
                                 "d_alpha",
                                 "d_beta",
                                 "b_trainingFromFile",
                                 "s_inputTrainingSet",
                                 "s_outputTrainingFile",
                                 "b_testingFromFile",
                                 "s_inputTestSet",
                                 "s_outputTestSet",
                                 "i_number_of_training",
                                 "i_trainingSetType",
                                 "i_outputTrainingSetType",
                                 "i_recurrent_flip_flop",
                                 "b_loadFromCheckpoint",
                                 "s_checkpoint_file",
                                 "b_loadNetworkFromFile",
                                 "s_loadNetworkFile",
                                 "i_numberTimesThroughFile",
                                 "Type:LongTermShortTerm_items",
                                 "i_backprop_unrolled",
                                 "i_number_in_sequence",
                                 "i_number_start_nodes",
                                 "i_number_new_weights",
                                 "i_number_of_testing_items",
                                 "b_allow_growth",
                                 "i_number_minutes_to_checkpoint"};

                if (Elements == null)
                {
                    createElementList();
                }
                string value;
              
                bool save_problem = false;
                for (int i = 0, k = 0; k < lines.Length; k++)
                {
                    if (!((string)lines[k]).StartsWith("Type") && i < Elements.Length)
                    {

                        if (Elements[i] is File_Select_Button)
                        {
                            lines[k]+=" " + ((File_Select_Button)Elements[i]).Text;
                        }
                        else if (Elements[i] is Number_Picker)
                        {
                            lines[k] += " " + ((Number_Picker)Elements[i]).Text;
                        }
                        else if (Elements[i] is True_False_Button)
                        {
                            lines[k] += " " + ((True_False_Button)Elements[i]).Text;
                        }
                        else if (Elements[i] is TextBox)
                        {
                            lines[k] += " "+((TextBox)Elements[i]).Text;

                        }

                        i++;
                    }
                    else if(Elements.Length <=i)
                    {
                        
                        save_problem = true;
                        break;
                    }
                }

                
                if (save_problem)
                {
                    if (MessageBox.Show("Issue In Code. Please Contact Developer. There might be some issues in the file. Would you like to save the file anyway?", "test", MessageBoxButton.YesNo) == MessageBoxResult.Yes)
                    {
                        System.IO.File.WriteAllLines(filename, lines);
                    }
                }
                else
                {
                    System.IO.File.WriteAllLines(filename, lines);
                }
            }
        }

        private void createElementList()
        {
            var value = _Grid_View.Children
                          .Cast<UIElement>()
                          .Where(i => (!(i is TextBox) || i is TextBox && ((TextBox)i).IsEnabled));
           this.Elements = value.ToArray();
        }
    }
}
