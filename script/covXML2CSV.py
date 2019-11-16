# #!/usr/bin/python
# import xml.etree.ElementTree as ET

# tree = ET.parse('../data/NYC_Subway_Stations.xml')
# root = tree.getroot()
# column_names = sorted(root[0].attrib.keys())

# print 'Data number  : %d'% len(root)
# print 'Data formate :', column_names
# #print 'Data tag : '+str(root.tag)

# new_csv = open('../data/NYC_Subway_Stations_xml.csv', 'w')
# for i, name in enumerate(column_names):
#     new_csv.write(name)
#     if i < len(column_names)-1:
#         new_csv.write(',')
#     else:
#         new_csv.write('\n')

# for branch in root:
#     for i, name in enumerate(column_names):
#         new_csv.write(branch.attrib[name].replace(',', ' '))
#         if i < len(column_names)-1:
#             new_csv.write(',')
#         else:
#             new_csv.write('\n')
# new_csv.close()
