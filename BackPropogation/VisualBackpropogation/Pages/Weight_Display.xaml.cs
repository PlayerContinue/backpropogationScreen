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

namespace VisualBackPropogation.Pages
{
    /// <summary>
    /// Interaction logic for Weight_Display.xaml
    /// </summary>
    public partial class Weight_Display : UserControl
    {
        public Weight_Display()
        {
            InitializeComponent();
        }

        public void add_weights(double weight)
        {
            _WeightGrid.Rows = _WeightGrid.Rows + 1;
            Button text = new Button();
            text.Content = weight.ToString();
            _WeightGrid.Children.Add(text);
        }
    }
}
