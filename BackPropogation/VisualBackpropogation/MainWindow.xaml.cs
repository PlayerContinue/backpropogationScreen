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

namespace VisualBackPropogation
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
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

        private void MenuItem_Click(object sender, RoutedEventArgs e)
        {
            Launch_Learning_Algorithm temp = new Launch_Learning_Algorithm();
            temp.Launch();
        }
    }
}
