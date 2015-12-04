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
    /// Interaction logic for Number_Picker.xaml
    /// </summary>
    public partial class Number_Picker : UserControl
    {
        [Description("Increase By Decimal Or Integer"), Category("Data"), Browsable(false)]
        String incrementValue;
        public string IncrementType
        {
            get{
                return incrementValue ?? "Integer";   
            }
            
            set
            {
                incrementValue = value;
            }
        }

        [Description("Increase By Decimal Or Integer"), Category("Data"), Browsable(false)]
        public string Text
        {
            get
            {
                return _Number_Value.Text;
            }
            set
            {
                _Number_Value.Text = value;
            }
        }


        public Number_Picker()
        {
            InitializeComponent();
          
        }

        private void Increment(object sender, RoutedEventArgs e)
        {
            Change_Value(1);
        }

        private void Decrement(object sender, RoutedEventArgs e)
        {
            Change_Value(-1);
        }

     
        private void Change_Value(int value)
        {
            if (IncrementType.CompareTo("Decimal")==0)
            {
                this.Text = (Double.Parse(this.Text) + (value*.1)).ToString();
            }
            else
            {
                this.Text = (Int32.Parse(this.Text) + (int)value).ToString();
            }

            
        }

    }
}
