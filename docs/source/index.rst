########################
rocALUTION Documentation
########################

rocALUTION is a sparse linear algebra library with focus on exploring fine-grained parallelism on top of AMD's Radeon Open Compute ROCm runtime and toolchains, targeting modern CPU and GPU platforms.
Based on C++ and HIP, it provides a portable, generic and flexible design that allows seamless integration with other scientific software packages.

In the following, three separate chapters are available:

  * :ref:`user_manual`: This is the manual of rocALUTION. It can be seen as a starting guide for new users but also a reference book for more experienced users.
  * :ref:`design_document`: The Design Document is targeted to advanced users / developers that want to understand, modify or extend the functionality of the rocALUTION library.
    To embed rocALUTION into your project, it is not required to read the Design Document.
  * :ref:`api`: This is a list of API functions provided by rocALUTION.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   usermanual
   designdoc
   api
