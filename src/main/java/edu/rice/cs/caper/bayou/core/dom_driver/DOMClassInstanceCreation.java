/*
Copyright 2017 Rice University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package edu.rice.cs.caper.bayou.core.dom_driver;


import edu.rice.cs.caper.bayou.core.dsl.DAPICall;
import edu.rice.cs.caper.bayou.core.dsl.DSubTree;
import org.eclipse.jdt.core.dom.*;

public class DOMClassInstanceCreation implements Handler {

    final ClassInstanceCreation creation;

    public DOMClassInstanceCreation(ClassInstanceCreation creation) {
        this.creation = creation;
    }

    @Override
    public DSubTree handle() {
        DSubTree tree = new DSubTree();
        // add the expression's subtree (e.g: foo(..).bar() should handle foo(..) first)
        DSubTree Texp = new DOMExpression(creation.getExpression()).handle();
        tree.addNodes(Texp.getNodes());

        // evaluate arguments first
        for (Object o : creation.arguments()) {
            DSubTree Targ = new DOMExpression((Expression) o).handle();
            tree.addNodes(Targ.getNodes());
        }

        IMethodBinding binding = creation.resolveConstructorBinding();

        // check if the binding is of a generic type that involves user-defined types
        if (binding != null) {
            ITypeBinding cls = binding.getDeclaringClass();
            boolean userType = false;
            if (cls != null && cls.isParameterizedType())
                for (int i = 0; i < cls.getTypeArguments().length; i++)
                    userType |= !cls.getTypeArguments()[i].getQualifiedName().startsWith("java.")
                            && !cls.getTypeArguments()[i].getQualifiedName().startsWith("javax.");

            if (userType || cls == null) // get to the generic declaration
                while (binding != null && binding.getMethodDeclaration() != binding)
                    binding = binding.getMethodDeclaration();
        }

        if (Utils.isRelevantCall(binding)) {
            try {
                tree.addNode(new DAPICall(binding));
            } catch (DAPICall.InvalidAPICallException e) {
                // continue without adding the node
            }
        }
        return tree;
    }
}
