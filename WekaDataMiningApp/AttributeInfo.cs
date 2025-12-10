using System;
using System.Collections.Generic;

namespace WekaDataMiningApp
{
    public class AttributeInfo
    {
        public string Name { get; set; }
        public bool IsNominal { get; set; }
        public List<string> PossibleValues { get; set; }
        
        public AttributeInfo()
        {
            PossibleValues = new List<string>();
        }
    }
}
