using System;
using System.Collections.Generic;
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
using VisualBackPropogation.Helper;

namespace VisualBackPropogation
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        String Loaded_Settings;
        VisualBackPropogation.Pages.Settings_Form Settings_Page;
        VisualBackPropogation.GraphView Graph;
        ICommand onMenuChangeScreensCommand;
        String Run_Location;//Store the location of the process to run
        public ICommand OnMenuChangeScreensCommand
        {
            get
            {
                return onMenuChangeScreensCommand ??
                    (onMenuChangeScreensCommand = new RelayCommand(change_screens));
            }
        }
        public MainWindow()
        {
            try
            {
                InitializeComponent();
                Settings_Page = new VisualBackPropogation.Pages.Settings_Form();
                _mainFrame.Navigate(Settings_Page);
            }
            catch (Exception ex)
            {
                ConsoleManager.Show();
                Console.Write(ex.Message);

            }
        }


        private void load_file(object sender, RoutedEventArgs e)
        {
            if (Settings_Page == null)
            {
                Settings_Page = new VisualBackPropogation.Pages.Settings_Form();
            }

            Loaded_Settings = Settings_Page.LoadFile();
            if (_mainFrame.Content is VisualBackPropogation.Pages.Settings_Form)
            {
                Settings_Page.Load_File_Info(Loaded_Settings);
            }
        }


        private void change_screens(object sender)
        {
            string temp = sender as String;
            switch (temp)
            {
                case "Settings_Page":

                    Switch_To_Settings();
                    break;

                case "Launch_Program":

                    Switch_To_Settings();
                    if (Run_Location == null)
                    {
                        this.Run_Location = this.LoadFileLocation();
                    }
                    if (Loaded_Settings == null)
                    {
                        this.Loaded_Settings = this.Settings_Page.LoadFile();
                    }

                    Settings_Page.Load_File_Info(this.Loaded_Settings);
                    Settings_Page.Launch_Learning_Algorithm(this.Run_Location);
                    break;

                case "Graph_View":
                    if (Graph == null)
                    {
                        this.Graph = new VisualBackPropogation.GraphView();
                    }
                    _mainFrame.Navigate(this.Graph);
                    break;
            }
        }

        private void Switch_To_Settings()
        {
            if (Settings_Page == null)
            {
                Settings_Page = new VisualBackPropogation.Pages.Settings_Form();
            }



            _mainFrame.Navigate(Settings_Page);

            if (Loaded_Settings != null)
            {
                Settings_Page.Load_File_Info(Loaded_Settings);
            }
        }


        private string LoadFileLocation()
        {
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();

            //Filter the file type
            dlg.DefaultExt = ".txt";
            dlg.Filter = "Application (*.exe)| *.exe";
            // Display OpenFileDialog by calling ShowDialog method 
            Nullable<bool> result = dlg.ShowDialog();
            if (result == true)
            {
                return dlg.FileName;
            }
            else
            {
                return null;
            }

        }

    }


}
