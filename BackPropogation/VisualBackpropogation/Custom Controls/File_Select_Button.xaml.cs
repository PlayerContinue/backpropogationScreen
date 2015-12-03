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
        public File_Select_Button()
        {
            InitializeComponent();
        }

      


        private void Open_File_Selector(object sender, RoutedEventArgs e)
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
    }
}
