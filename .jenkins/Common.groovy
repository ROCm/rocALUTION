// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project)
{
    project.paths.construct_build_prefix()
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=${project.compiler.compiler_path} ${project.paths.build_command}
            """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocalution-test --gtest_output=xml --gtest_color=yes #--gtest_filter=${gfilter}-*known_bug*
                """

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
}

def runPackageCommand(platform, project)
{
    def command

    if(platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles'))
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                mkdir -p package
                mv *.rpm package/
                rpm -qlp package/*.rpm
            """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")
    }
    else if(platform.jenkinsLabel.contains('hip-clang'))
    {
        packageCommand = null
    }
    else
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                mkdir -p package
                mv *.deb package/
                dpkg -c package/*.deb
            """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
    }
}

echo "Common Groovy Loaded"

return this

