#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('30 22 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])


////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rocALUTIONCIhost:
{

    def rocalution = new rocProject('rocalution_host')
    // customize for project
    rocalution.paths.build_command = './install.sh -c --host'
    rocalution.compiler.compiler_name = 'c++'
    rocalution.compiler.compiler_path = 'c++'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['dkms'], rocalution)

    boolean formatCheck = true

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        def command = """#!/usr/bin/env bash
                  set -x
                  cd ${project.paths.project_build_prefix}
                  LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=${project.compiler.compiler_path} ${project.paths.build_command}
                """

        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        String sudo = auxiliary.sudo(platform.jenkinsLabel)
        def gfilter = auxiliary.isJobStartedByTimer() ? "*nightly*" : "*checkin*"
        def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/release/clients/staging
                        ${sudo} LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocalution-test --gtest_output=xml --gtest_color=yes #--gtest_filter=${gfilter}-*known_bug*
                    """

        platform.runCommand(this, command)
        junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
    }

    def packageCommand =
    {
        platform, project->

        def command = """
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

    buildProject(rocalution, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}

rocALUTIONCImpi:
{

    def rocalution = new rocProject('rocalution_mpi')
    // customize for project
    rocalution.paths.build_command = './install.sh -c --host --mpi --no-openmp'
    rocalution.compiler.compiler_name = 'c++'
    rocalution.compiler.compiler_path = 'c++'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['dkms'], rocalution)

    boolean formatCheck = true

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        def command = """#!/usr/bin/env bash
                  set -x
                  cd ${project.paths.project_build_prefix}
                  LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=${project.compiler.compiler_path} ${project.paths.build_command}
                """

        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        String sudo = auxiliary.sudo(platform.jenkinsLabel)
        def gfilter = auxiliary.isJobStartedByTimer() ? "*nightly*" : "*checkin*"
        def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/release/clients/staging
                        ${sudo} LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocalution-test --gtest_output=xml --gtest_color=yes #--gtest_filter=${gfilter}-*known_bug*
                    """

        platform.runCommand(this, command)
        junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
    }

    def packageCommand =
    {
        platform, project->

        def command = """
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

    buildProject(rocalution, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}

rocALUTIONCI:
{

    def rocalution = new rocProject('rocALUTION')
    // customize for project
    rocalution.paths.build_command = './install.sh -c'
    rocalution.compiler.compiler_name = 'c++'
    rocalution.compiler.compiler_path = 'c++'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx900 && ubuntu', 'gfx906 && centos7', 'gfx906 && sles'], rocalution)

    boolean formatCheck = true

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=${project.compiler.compiler_path} ${project.paths.build_command}
                """

        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        String sudo = auxiliary.sudo(platform.jenkinsLabel)
        def gfilter = auxiliary.isJobStartedByTimer() ? "*nightly*" : "*checkin*"
        def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/release/clients/staging
                        ${sudo} LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocalution-test --gtest_output=xml --gtest_color=yes #--gtest_filter=${gfilter}-*known_bug*
                    """

        platform.runCommand(this, command)
        junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
    }

    def packageCommand =
    {
        platform, project->

        def command

        if(platform.jenkinsLabel.contains('centos'))
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
        else if(platform.jenkinsLabel.contains('hip-clang') || platform.jenkinsLabel.contains('sles'))
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

    buildProject(rocalution, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}
