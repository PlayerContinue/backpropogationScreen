using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace VisualBackPropogation.Custom_Controls
{
    /// <summary>
    /// Interaction logic for File_Select_Button.xaml
    /// </summary>
    public partial class File_Select_Button : UserControl
    {
        private string save_open="Open";
        [Description("File Information"), Category("Data"), Browsable(false)]
        public String Text
        {
            get
            {
                return File_Name.Text;
            }
            set
            {
                File_Name.Text = value;
            }
        }

        [Description("File Information"), Category("Data"), Browsable(false)]
        public String Save_Open{
            get{
                return save_open;
            }

            set{
                save_open = value;
            }
        }

        public File_Select_Button()
        {
            InitializeComponent();
        }

        private void Open_Selector()
        {
            // Create OpenFileDialog 
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();



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
                this.Text = filename;
            }
        }

        private void Open_Save_Selector()
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
                this.Text = filename;
            }
        }

        private void Open_File_Selector(object sender, RoutedEventArgs e)
        {
            if (save_open.CompareTo("Save") == 0)
            {
                Open_Selector();
            }
            else
            {
                Open_Save_Selector();
            }
        }
    }
}
