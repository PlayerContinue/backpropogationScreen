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
        ICommand onMenuChangeScreensCommand;
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
                GraphView temp = new GraphView();
                _mainFrame.Navigate(temp);
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

            Loaded_Settings= Settings_Page.LoadFile();
        }
      

        private void change_screens(object sender)
        {
            string temp = sender as String;
            switch (temp)
            {
                case "Settings_Page":
                    if (Settings_Page == null)
                    {
                        Settings_Page = new VisualBackPropogation.Pages.Settings_Form();
                    }

                  

                    _mainFrame.Navigate(Settings_Page);

                    if (Loaded_Settings != null)
                    {
                        Settings_Page.Load_File_Info(Loaded_Settings);
                    }

                    break;
                case "Graph_View":
                    _mainFrame.Navigate(new GraphView());
                    break;
            }
        }

        private void MenuItem_Click(object sender, RoutedEventArgs e)
        {
            Launch_Learning_Algorithm temp = new Launch_Learning_Algorithm();
            temp.Launch();
        }
    }
}
