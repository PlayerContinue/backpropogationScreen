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
    /// Interaction logic for GraphView.xaml
    /// </summary>
    public partial class GraphView : Page
    {
        public GraphView()
        {
            try
            {
                InitializeComponent();
                
            }
            catch (Exception ex)
            {
                ConsoleManager.Show();
                Console.Write(ex.Message);
            }

            List<int[]> temp = new List<int[]>();
            Random rand = new Random();
            for (int i = 0; i < 10; i++)
            {
                temp.Add(new int[2] {rand.Next(10), rand.Next(10) });
                _WeightGrid.add_weights(rand.NextDouble());
            }
            temp.Add(new int[2] { 0, 0 });
                _MainGraphView.DrawGraph(temp,10);
                this.AddHandler(Button.ClickEvent, new RoutedEventHandler(BubbelingWeightClick));
           
        }

       
       
        private void BubbelingWeightClick(object o, RoutedEventArgs e)
        {
            VisualBackPropogation.Pages.Weight_Display pressed = e.Source as VisualBackPropogation.Pages.Weight_Display;

            MessageBox.Show(pressed.Content.ToString());
        }
    

        
    }
}
