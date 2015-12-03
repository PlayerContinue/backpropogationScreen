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

namespace VisualBackPropogation.Custom_Controls
{
    /// <summary>
    /// Interaction logic for True_False_Button.xaml
    /// </summary>
    public partial class True_False_Button : UserControl
    {
        private static int count =0;


        String groupName;
        [System.ComponentModel.Description("Group Name"), System.ComponentModel.Category("Data"), System.ComponentModel.Browsable(false)]
        public string GroupName
        {
            get
            {
                return groupName ?? "Group1";
            }

            set
            {
                groupName = value;
            }
        }
         [System.ComponentModel.Description("Set True or False"), System.ComponentModel.Category("Data"), System.ComponentModel.Browsable(false)]
        public bool? IsChecked
        {
            get
            {
                return TrueButton.IsChecked;
            }

            set
            {
                TrueButton.IsChecked = value;
            }
        }

        [System.ComponentModel.Description("Set the Visible Value"), System.ComponentModel.Category("Data"), System.ComponentModel.Browsable(false)]
         public string Text
         {
             get
             {
                 if (TrueButton.IsChecked.Value)
                 {
                     return "1";
                 }
                 else
                 {
                     return "0";
                 }
             }
             set
             {
                 if (value == "1")
                 {
                     TrueButton.IsChecked = true;
                     FalseButton.IsChecked = false;
                 }
                 else
                 {
                     FalseButton.IsChecked = true;
                     TrueButton.IsChecked = false;
                 }
             }
         }

      

        public True_False_Button()
        {
            count++;
            groupName = "Group" + count;
            InitializeComponent();
           
        }
    }
}
